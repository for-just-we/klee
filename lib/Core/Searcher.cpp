//===-- Searcher.cpp ------------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Searcher.h"

#include "CoreStats.h"
#include "ExecutionState.h"
#include "Executor.h"
#include "MergeHandler.h"
#include "PTree.h"
#include "StatsTracker.h"

#include "klee/ADT/DiscretePDF.h"
#include "klee/ADT/RNG.h"
#include "klee/Statistics/Statistics.h"
#include "klee/Module/InstructionInfoTable.h"
#include "klee/Module/KInstruction.h"
#include "klee/Module/KModule.h"
#include "klee/Support/ErrorHandling.h"
#include "klee/System/Time.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

#include <Python.h>

#include <cassert>
#include <cmath>

using namespace klee;
using namespace llvm;


///
ExecutionState &DFSSearcher::selectState() {
  return *states.back();
}

void DFSSearcher::update(ExecutionState *current,
                         const std::vector<ExecutionState *> &addedStates,
                         const std::vector<ExecutionState *> &removedStates) {
  // insert states
  states.insert(states.end(), addedStates.begin(), addedStates.end());

  // remove states
  for (const auto state : removedStates) {
    if (state == states.back()) {
      states.pop_back();
    } else {
      auto it = std::find(states.begin(), states.end(), state);
      assert(it != states.end() && "invalid state removed");
      states.erase(it);
    }
  }
}

bool DFSSearcher::empty() {
  return states.empty();
}

void DFSSearcher::printName(llvm::raw_ostream &os) {
  os << "DFSSearcher\n";
}


///

ExecutionState &BFSSearcher::selectState() {
  return *states.front();
}

void BFSSearcher::update(ExecutionState *current,
                         const std::vector<ExecutionState *> &addedStates,
                         const std::vector<ExecutionState *> &removedStates) {
  // update current state
  // Assumption: If new states were added KLEE forked, therefore states evolved.
  // constraints were added to the current state, it evolved.
  if (!addedStates.empty() && current &&
      std::find(removedStates.begin(), removedStates.end(), current) == removedStates.end()) {
    auto pos = std::find(states.begin(), states.end(), current);
    assert(pos != states.end());
    states.erase(pos);
    states.push_back(current);
  }

  // insert states
  states.insert(states.end(), addedStates.begin(), addedStates.end());

  // remove states
  for (const auto state : removedStates) {
    if (state == states.front()) {
      states.pop_front();
    } else {
      auto it = std::find(states.begin(), states.end(), state);
      assert(it != states.end() && "invalid state removed");
      states.erase(it);
    }
  }
}

bool BFSSearcher::empty() {
  return states.empty();
}

void BFSSearcher::printName(llvm::raw_ostream &os) {
  os << "BFSSearcher\n";
}


///

RandomSearcher::RandomSearcher(RNG &rng) : theRNG{rng} {}

ExecutionState &RandomSearcher::selectState() {
  return *states[theRNG.getInt32() % states.size()];
}

void RandomSearcher::update(ExecutionState *current,
                            const std::vector<ExecutionState *> &addedStates,
                            const std::vector<ExecutionState *> &removedStates) {
  // insert states
  states.insert(states.end(), addedStates.begin(), addedStates.end());

  // remove states
  for (const auto state : removedStates) {
    auto it = std::find(states.begin(), states.end(), state);
    assert(it != states.end() && "invalid state removed");
    states.erase(it);
  }
}

bool RandomSearcher::empty() {
  return states.empty();
}

void RandomSearcher::printName(llvm::raw_ostream &os) {
  os << "RandomSearcher\n";
}


///

WeightedRandomSearcher::WeightedRandomSearcher(WeightType type, RNG &rng)
  : states(std::make_unique<DiscretePDF<ExecutionState*, ExecutionStateIDCompare>>()),
    theRNG{rng},
    type(type) {

  switch(type) {
  case Depth:
  case RP:
    updateWeights = false;
    break;
  case InstCount:
  case CPInstCount:
  case QueryCost:
  case MinDistToUncovered:
  case CoveringNew:
    updateWeights = true;
    break;
  default:
    assert(0 && "invalid weight type");
  }
}

ExecutionState &WeightedRandomSearcher::selectState() {
  return *states->choose(theRNG.getDoubleL());
}

double WeightedRandomSearcher::getWeight(ExecutionState *es) {
  switch(type) {
    default:
    case Depth:
      return es->depth;
    case RP:
      return std::pow(0.5, es->depth);
    case InstCount: {
      uint64_t count = theStatisticManager->getIndexedValue(stats::instructions,
                                                            es->pc->info->id);
      double inv = 1. / std::max((uint64_t) 1, count);
      return inv * inv;
    }
    case CPInstCount: {
      StackFrame &sf = es->stack.back();
      uint64_t count = sf.callPathNode->statistics.getValue(stats::instructions);
      double inv = 1. / std::max((uint64_t) 1, count);
      return inv;
    }
    case QueryCost:
      return (es->queryMetaData.queryCost.toSeconds() < .1)
                 ? 1.
                 : 1. / es->queryMetaData.queryCost.toSeconds();
    case CoveringNew:
    case MinDistToUncovered: {
      uint64_t md2u = computeMinDistToUncovered(es->pc,
                                                es->stack.back().minDistToUncoveredOnReturn);

      double invMD2U = 1. / (md2u ? md2u : 10000);
      if (type == CoveringNew) {
        double invCovNew = 0.;
        if (es->instsSinceCovNew)
          invCovNew = 1. / std::max(1, (int) es->instsSinceCovNew - 1000);
        return (invCovNew * invCovNew + invMD2U * invMD2U);
      } else {
        return invMD2U * invMD2U;
      }
    }
  }
}

void WeightedRandomSearcher::update(ExecutionState *current,
                                    const std::vector<ExecutionState *> &addedStates,
                                    const std::vector<ExecutionState *> &removedStates) {

  // update current
  if (current && updateWeights &&
      std::find(removedStates.begin(), removedStates.end(), current) == removedStates.end())
    states->update(current, getWeight(current));

  // insert states
  for (const auto state : addedStates)
    states->insert(state, getWeight(state));

  // remove states
  for (const auto state : removedStates)
    states->remove(state);
}

bool WeightedRandomSearcher::empty() {
  return states->empty();
}

void WeightedRandomSearcher::printName(llvm::raw_ostream &os) {
  os << "WeightedRandomSearcher::";
  switch(type) {
    case Depth              : os << "Depth\n"; return;
    case RP                 : os << "RandomPath\n"; return;
    case QueryCost          : os << "QueryCost\n"; return;
    case InstCount          : os << "InstCount\n"; return;
    case CPInstCount        : os << "CPInstCount\n"; return;
    case MinDistToUncovered : os << "MinDistToUncovered\n"; return;
    case CoveringNew        : os << "CoveringNew\n"; return;
    default                 : os << "<unknown type>\n"; return;
  }
}


///

// Check if n is a valid pointer and a node belonging to us
#define IS_OUR_NODE_VALID(n)                                                   \
  (((n).getPointer() != nullptr) && (((n).getInt() & idBitMask) != 0))

RandomPathSearcher::RandomPathSearcher(PTree &processTree, RNG &rng)
  : processTree{processTree},
    theRNG{rng},
    idBitMask{processTree.getNextId()} {};

ExecutionState &RandomPathSearcher::selectState() {
  unsigned flips=0, bits=0;
  assert(processTree.root.getInt() & idBitMask && "Root should belong to the searcher");
  PTreeNode *n = processTree.root.getPointer();
  while (!n->state) {
    if (!IS_OUR_NODE_VALID(n->left)) {
      assert(IS_OUR_NODE_VALID(n->right) && "Both left and right nodes invalid");
      assert(n != n->right.getPointer());
      n = n->right.getPointer();
    } else if (!IS_OUR_NODE_VALID(n->right)) {
      assert(IS_OUR_NODE_VALID(n->left) && "Both right and left nodes invalid");
      assert(n != n->left.getPointer());
      n = n->left.getPointer();
    } else {
      if (bits==0) {
        flips = theRNG.getInt32();
        bits = 32;
      }
      --bits;
      n = ((flips & (1U << bits)) ? n->left : n->right).getPointer();
    }
  }

  return *n->state;
}

void RandomPathSearcher::update(ExecutionState *current,
                                const std::vector<ExecutionState *> &addedStates,
                                const std::vector<ExecutionState *> &removedStates) {
  // insert states
  for (auto es : addedStates) {
    PTreeNode *pnode = es->ptreeNode, *parent = pnode->parent;
    PTreeNodePtr *childPtr;

    childPtr = parent ? ((parent->left.getPointer() == pnode) ? &parent->left
                                                              : &parent->right)
                      : &processTree.root;
    while (pnode && !IS_OUR_NODE_VALID(*childPtr)) {
      childPtr->setInt(childPtr->getInt() | idBitMask);
      pnode = parent;
      if (pnode)
        parent = pnode->parent;

      childPtr = parent
                     ? ((parent->left.getPointer() == pnode) ? &parent->left
                                                             : &parent->right)
                     : &processTree.root;
    }
  }

  // remove states
  for (auto es : removedStates) {
    PTreeNode *pnode = es->ptreeNode, *parent = pnode->parent;

    while (pnode && !IS_OUR_NODE_VALID(pnode->left) &&
           !IS_OUR_NODE_VALID(pnode->right)) {
      auto childPtr =
          parent ? ((parent->left.getPointer() == pnode) ? &parent->left
                                                         : &parent->right)
                 : &processTree.root;
      assert(IS_OUR_NODE_VALID(*childPtr) && "Removing pTree child not ours");
      childPtr->setInt(childPtr->getInt() & ~idBitMask);
      pnode = parent;
      if (pnode)
        parent = pnode->parent;
    }
  }
}

bool RandomPathSearcher::empty() {
  return !IS_OUR_NODE_VALID(processTree.root);
}

void RandomPathSearcher::printName(llvm::raw_ostream &os) {
  os << "RandomPathSearcher\n";
}


///

MergingSearcher::MergingSearcher(Searcher *baseSearcher)
  : baseSearcher{baseSearcher} {};

void MergingSearcher::pauseState(ExecutionState &state) {
  assert(std::find(pausedStates.begin(), pausedStates.end(), &state) == pausedStates.end());
  pausedStates.push_back(&state);
  baseSearcher->update(nullptr, {}, {&state});
}

void MergingSearcher::continueState(ExecutionState &state) {
  auto it = std::find(pausedStates.begin(), pausedStates.end(), &state);
  assert(it != pausedStates.end());
  pausedStates.erase(it);
  baseSearcher->update(nullptr, {&state}, {});
}

ExecutionState& MergingSearcher::selectState() {
  assert(!baseSearcher->empty() && "base searcher is empty");

  if (!UseIncompleteMerge)
    return baseSearcher->selectState();

  // Iterate through all MergeHandlers
  for (auto cur_mergehandler: mergeGroups) {
    // Find one that has states that could be released
    if (!cur_mergehandler->hasMergedStates()) {
      continue;
    }
    // Find a state that can be prioritized
    ExecutionState *es = cur_mergehandler->getPrioritizeState();
    if (es) {
      return *es;
    } else {
      if (DebugLogIncompleteMerge){
        llvm::errs() << "Preemptively releasing states\n";
      }
      // If no state can be prioritized, they all exceeded the amount of time we
      // are willing to wait for them. Release the states that already arrived at close_merge.
      cur_mergehandler->releaseStates();
    }
  }
  // If we were not able to prioritize a merging state, just return some state
  return baseSearcher->selectState();
}

void MergingSearcher::update(ExecutionState *current,
                             const std::vector<ExecutionState *> &addedStates,
                             const std::vector<ExecutionState *> &removedStates) {
  // We have to check if the current execution state was just deleted, as to
  // not confuse the nurs searchers
  if (std::find(pausedStates.begin(), pausedStates.end(), current) == pausedStates.end()) {
    baseSearcher->update(current, addedStates, removedStates);
  }
}

bool MergingSearcher::empty() {
  return baseSearcher->empty();
}

void MergingSearcher::printName(llvm::raw_ostream &os) {
  os << "MergingSearcher\n";
}


///

BatchingSearcher::BatchingSearcher(Searcher *baseSearcher, time::Span timeBudget, unsigned instructionBudget)
  : baseSearcher{baseSearcher},
    timeBudget{timeBudget},
    instructionBudget{instructionBudget} {};

ExecutionState &BatchingSearcher::selectState() {
  if (!lastState ||
      (((timeBudget.toSeconds() > 0) &&
        (time::getWallTime() - lastStartTime) > timeBudget)) ||
      ((instructionBudget > 0) &&
       (stats::instructions - lastStartInstructions) > instructionBudget)) {
    if (lastState) {
      time::Span delta = time::getWallTime() - lastStartTime;
      auto t = timeBudget;
      t *= 1.1;
      if (delta > t) {
        klee_message("increased time budget from %f to %f\n", timeBudget.toSeconds(), delta.toSeconds());
        timeBudget = delta;
      }
    }
    lastState = &baseSearcher->selectState();
    lastStartTime = time::getWallTime();
    lastStartInstructions = stats::instructions;
    return *lastState;
  } else {
    return *lastState;
  }
}

void BatchingSearcher::update(ExecutionState *current,
                              const std::vector<ExecutionState *> &addedStates,
                              const std::vector<ExecutionState *> &removedStates) {
  // drop memoized state if it is marked for deletion
  if (std::find(removedStates.begin(), removedStates.end(), lastState) != removedStates.end())
    lastState = nullptr;
  // update underlying searcher
  baseSearcher->update(current, addedStates, removedStates);
}

bool BatchingSearcher::empty() {
  return baseSearcher->empty();
}

void BatchingSearcher::printName(llvm::raw_ostream &os) {
  os << "<BatchingSearcher> timeBudget: " << timeBudget
     << ", instructionBudget: " << instructionBudget
     << ", baseSearcher:\n";
  baseSearcher->printName(os);
  os << "</BatchingSearcher>\n";
}


///

IterativeDeepeningTimeSearcher::IterativeDeepeningTimeSearcher(Searcher *baseSearcher)
  : baseSearcher{baseSearcher} {};

ExecutionState &IterativeDeepeningTimeSearcher::selectState() {
  ExecutionState &res = baseSearcher->selectState();
  startTime = time::getWallTime();
  return res;
}

void IterativeDeepeningTimeSearcher::update(ExecutionState *current,
                                            const std::vector<ExecutionState *> &addedStates,
                                            const std::vector<ExecutionState *> &removedStates) {

  const auto elapsed = time::getWallTime() - startTime;

  // update underlying searcher (filter paused states unknown to underlying searcher)
  if (!removedStates.empty()) {
    std::vector<ExecutionState *> alt = removedStates;
    for (const auto state : removedStates) {
      auto it = pausedStates.find(state);
      if (it != pausedStates.end()) {
        pausedStates.erase(it);
        alt.erase(std::remove(alt.begin(), alt.end(), state), alt.end());
      }
    }    
    baseSearcher->update(current, addedStates, alt);
  } else {
    baseSearcher->update(current, addedStates, removedStates);
  }

  // update current: pause if time exceeded
  if (current &&
      std::find(removedStates.begin(), removedStates.end(), current) == removedStates.end() &&
      elapsed > time) {
    pausedStates.insert(current);
    baseSearcher->update(nullptr, {}, {current});
  }

  // no states left in underlying searcher: fill with paused states
  if (baseSearcher->empty()) {
    time *= 2U;
    klee_message("increased time budget to %f\n", time.toSeconds());
    std::vector<ExecutionState *> ps(pausedStates.begin(), pausedStates.end());
    baseSearcher->update(nullptr, ps, std::vector<ExecutionState *>());
    pausedStates.clear();
  }
}

bool IterativeDeepeningTimeSearcher::empty() {
  return baseSearcher->empty() && pausedStates.empty();
}

void IterativeDeepeningTimeSearcher::printName(llvm::raw_ostream &os) {
  os << "IterativeDeepeningTimeSearcher\n";
}

///

InterleavedSearcher::InterleavedSearcher(const std::vector<Searcher*> &_searchers) {
  searchers.reserve(_searchers.size());
  for (auto searcher : _searchers)
    searchers.emplace_back(searcher);
}

ExecutionState &InterleavedSearcher::selectState() {
  Searcher *s = searchers[--index].get();
  if (index == 0) index = searchers.size();
  return s->selectState();
}

void InterleavedSearcher::update(ExecutionState *current,
                                 const std::vector<ExecutionState *> &addedStates,
                                 const std::vector<ExecutionState *> &removedStates) {

  // update underlying searchers
  for (auto &searcher : searchers)
    searcher->update(current, addedStates, removedStates);
}

bool InterleavedSearcher::empty() {
  return searchers[0]->empty();
}

void InterleavedSearcher::printName(llvm::raw_ostream &os) {
  os << "<InterleavedSearcher> containing " << searchers.size() << " searchers:\n";
  for (const auto &searcher : searchers)
    searcher->printName(os);
  os << "</InterleavedSearcher>\n";
}

// Add support for subpath guided search
SubpathGuidedSearcher::SubpathGuidedSearcher(Executor &_executor, uint index)
    : executor(_executor), index(index),theRNG{_executor.theRNG} {
}

ExecutionState &SubpathGuidedSearcher::selectState() {
  unsigned long minCount = ULONG_MAX;
  std::vector<ExecutionState *> selectSet;
  // std::cout << "states:" << std::endl;
  for(auto & state : states) {
    subpath_ty subpath;
    executor.getSubpath(state, subpath, index);
    unsigned long curr = executor.getSubpathCount(subpath, index);
    if(curr < minCount) {
      selectSet.clear();
      minCount = curr;
    }

    if(curr == minCount) {
      selectSet.push_back(state);
    }
  }

  unsigned int random = theRNG.getInt32() % selectSet.size();
  ExecutionState *selection = selectSet[random];
  // P[ES.π]++
  subpath_ty subpath;
  executor.getSubpath(selection, subpath, index);
  executor.incSubpath(subpath, index);

  return *selection;
}

void SubpathGuidedSearcher::update(ExecutionState *current,
                                   const std::vector<ExecutionState *> &addedStates,
                                   const std::vector<ExecutionState *> &removedStates) {
  states.insert(states.end(),
                addedStates.begin(),
                addedStates.end());
  for (const auto state : removedStates) {
    auto it = std::find(states.begin(), states.end(), state);
    assert(it != states.end() && "invalid state removed");
    states.erase(it);
  }
}


// add support for learch machine learning based search
MLSearcher::MLSearcher(Executor &_executor, std::string model_type, std::string model_path,
                       std::string scirpt_path, std::string py_path, bool _sampling) :
    executor(_executor), sampling(_sampling) {
    // execute init_model in model.py
    // 设置python解释器路径，默认会使用环境变量中的
    if (py_path.empty()){
        wchar_t* _py_path = new wchar_t[py_path.size() + 1];
        swprintf(_py_path, py_path.size() + 1, L"%s", py_path.c_str());
        Py_SetPythonHome(_py_path);
        klee_message("KLEE: using python interpreter: %s\n", py_path.c_str());
    }
    // 初始化python解释器.C/C++中调用Python之前必须先初始化解释器
    Py_Initialize();
    // lock
    PyGILState_STATE gstate = PyGILState_Ensure();
    // 2、初始化python系统文件路径，保证可以访问到 .py文件
    PyRun_SimpleString("import sys");
    std::string code = "sys.path.append('";
    code += scirpt_path;
    code += "')";
    PyRun_SimpleString(code.c_str());
    // 初始化使用的变量
    klee_message("KLEE: load python module: %s/model.py", scirpt_path.c_str());
    PyObject *pName = PyUnicode_FromString("model"); // import from model.py
    PyObject *pModule = PyImport_Import(pName); // representing modules in model.py
    if (pModule == NULL)
        klee_error("unable to load model.py");

    if (model_type == "linear")
        type = Linear;
    else if (model_type == "feedforward")
        type = Feedforward;
    else if (model_type == "rnn")
        type = RNN;
    // all init_model function in learch/model.py to load corresponding model
    klee_message("KLEE: load pytorch model: %s", model_path.c_str());
    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyBytes_FromString(model_type.c_str()));
    PyTuple_SetItem(pArgs, 1, PyBytes_FromString(model_path.c_str()));
    klee_message("KLEE: execute init_model function in model.py");
    PyObject* pInitFunc = PyObject_GetAttrString(pModule, "init_model"); // load init_model function
    PyObject_CallObject(pInitFunc, pArgs);
    klee_message("KLEE: init_model done\n");

    Py_DECREF(pModule);
    Py_DECREF(pName);
    Py_DECREF(pArgs);
    Py_DECREF(pInitFunc);
    // release lock
    PyGILState_Release(gstate);
}

MLSearcher::~MLSearcher() {
    // 结束python接口初始化
    klee_message("finalize python interpreter");
    Py_Finalize();
}

ExecutionState &MLSearcher::selectState() {
    PyGILState_STATE gstate = PyGILState_Ensure();
    int batch_size = 0;
    // features = list(), shape = [num_state, feature_size], the input to model, an element is a feature of a single state
    // hiddens = list(), and is only used in RNN model
    PyObject *features = PyList_New(0), *hiddens = PyList_New(0);
    // prepare input data to model, shape = [num_state, feature_size]
    for (ExecutionState* state : states) {
        ++batch_size;
        PyObject* feature = PyList_New(0); // feature for a single state
        executor.getStateFeatures(state);
        for (uint i=2; i<state->feature.size(); i++)
            PyList_Append(feature, PyFloat_FromDouble(state->feature[i])); // feature.append(state->feature[i])
        PyList_Append(features, feature); // features.append(feature)
        if (type == RNN) {
            PyObject* hidden = PyList_New(0); // hidden = list()
            for (uint i=0; i<state->hidden_state.size(); i++)  // for i in range(state.hidden_state)
                PyList_Append(hidden, PyFloat_FromDouble(state->hidden_state[i])); // hidden.append(state.hidden_state[i])
            PyList_Append(hiddens, hidden); // hiddens.append(hidden)
        }
    }

    if (batch_size > 0) {
        PyObject* pArgs = PyTuple_New(2);
        PyTuple_SetItem(pArgs, 0, features);
        PyTuple_SetItem(pArgs, 1, hiddens);
        PyObject *pName = PyUnicode_FromString("model");
        PyObject *pModule = PyImport_Import(pName);
        if (pModule == NULL)
            klee_error("unable to load model.py");
        // klee_message("KLEE: predict states");
        PyObject *pCallFunc = PyObject_GetAttrString(pModule, "predict");
        // call function predict of model.py
        PyObject* res = PyObject_CallObject(pCallFunc, pArgs);
        // klee_message("KLEE: predict done");
        PyObject* rewards = PyTuple_GetItem(res, 0); // rewards for each state, shape = [num_state]
        PyObject* new_hiddens = PyTuple_GetItem(res, 1); // only useful in RNN model
        // klee_message("KLEE: get item done");
        int i=0;
        for (auto state : states) {
            klee_message("KLEE: set attribute for state %d", i);
            state->predicted_reward = PyFloat_AsDouble(PyList_GetItem(rewards, i));
            if (type == RNN) {
                for (uint j=0; j<state->hidden_state.size(); j++)
                     state->hidden_state[j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(new_hiddens, i), j));
            }
            i++;
        }
    }

    ExecutionState *selection = NULL;
    // sampling为true表示采用随机算则，不过权重越大状态被选择的概率越大
    // klee_message("KLEE: select state");
    if (sampling) {
        PyObject* pArgs = PyTuple_New(1);
        PyObject* predicted = PyList_New(0); // predicted = list()
        for (auto state : states) {
            PyList_Append(predicted, PyFloat_FromDouble(state->predicted_reward)); // predicted.append(state->predicted_reward)
        }
        PyTuple_SetItem(pArgs, 0, predicted);
        // load model.py
        PyObject *pName = PyUnicode_FromString("model");
        PyObject *pModule = PyImport_Import(pName);
        // call sample function in model.py
        PyObject *pCallFunc = PyObject_GetAttrString(pModule, "sample");
        PyObject *res = PyObject_CallObject(pCallFunc, pArgs);
        selection = states[PyLong_AsLong(res)];
    }
    else {
        double current_max = -100000000.0;
        bool current_set = false;
        for (auto state : states) {
            // std::cout << state->predicted_reward << " ";
            if(!current_set || current_max < state->predicted_reward) {
                selection = state;
                current_max = state->predicted_reward;
                current_set = true;
            }
        }
    }
    selection->predicted_reward = 0.0;
    PyGILState_Release(gstate);
    addFeature(selection);
    // process features, which is post process of ML searh
    subpath_ty subpath;
    executor.getSubpath(selection, subpath, 0);
    executor.incSubpath(subpath, 0);
    executor.getSubpath(selection, subpath, 1);
    executor.incSubpath(subpath, 1);
    executor.getSubpath(selection, subpath, 2);
    executor.incSubpath(subpath, 2);
    executor.getSubpath(selection, subpath, 3);
    executor.incSubpath(subpath, 3);

    return *selection;
}

void MLSearcher::addFeature(ExecutionState *state){
    long index = featureIndex++;
    state->features.emplace_back(index, state->feature);
}

void MLSearcher::update(klee::ExecutionState *current,
                        const std::vector<ExecutionState *> &addedStates,
                        const std::vector<ExecutionState *> &removedStates) {
    states.insert(states.end(),
                  addedStates.begin(),
                  addedStates.end());
    for (const auto state : removedStates) {
        auto it = std::find(states.begin(), states.end(), state);
        assert(it != states.end() && "invalid state removed");
        states.erase(it);
    }
}

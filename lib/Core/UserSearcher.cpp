//===-- UserSearcher.cpp --------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UserSearcher.h"

#include "Executor.h"
#include "MergeHandler.h"
#include "Searcher.h"

#include "klee/Support/ErrorHandling.h"

#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace klee;

namespace {
llvm::cl::OptionCategory
    SearchCat("Search options", "These options control the search heuristic.");

cl::list<Searcher::CoreSearchType> CoreSearch(
    "search",
    cl::desc("Specify the search heuristic (default=random-path interleaved "
             "with nurs:covnew)"),
    cl::values(
        clEnumValN(Searcher::DFS, "dfs", "use Depth First Search (DFS)"),
        clEnumValN(Searcher::BFS, "bfs",
                   "use Breadth First Search (BFS), where scheduling decisions "
                   "are taken at the level of (2-way) forks"),
        clEnumValN(Searcher::RandomState, "random-state",
                   "randomly select a state to explore"),
        clEnumValN(Searcher::RandomPath, "random-path",
                   "use Random Path Selection (see OSDI'08 paper)"),
        clEnumValN(Searcher::NURS_CovNew, "nurs:covnew",
                   "use Non Uniform Random Search (NURS) with Coverage-New"),
        clEnumValN(Searcher::NURS_MD2U, "nurs:md2u",
                   "use NURS with Min-Dist-to-Uncovered"),
        clEnumValN(Searcher::NURS_Depth, "nurs:depth", "use NURS with depth"),
        clEnumValN(Searcher::NURS_RP, "nurs:rp", "use NURS with 1/2^depth"),
        clEnumValN(Searcher::NURS_ICnt, "nurs:icnt",
                   "use NURS with Instr-Count"),
        clEnumValN(Searcher::NURS_CPICnt, "nurs:cpicnt",
                   "use NURS with CallPath-Instr-Count"),
        clEnumValN(Searcher::NURS_QC, "nurs:qc", "use NURS with Query-Cost"),
        clEnumValN(Searcher::SGS_1, "sgs:1", "length 1 - Subpath-Guided Search"),
        clEnumValN(Searcher::SGS_2, "sgs:2", "length 2 - Subpath-Guided Search"),
        clEnumValN(Searcher::SGS_4, "sgs:4", "length 4 - Subpath-Guided Search"),
        clEnumValN(Searcher::SGS_8, "sgs:8", "length 8 - Subpath-Guided Search"),
        clEnumValN(Searcher::ML, "ml", "Machine Learning Search")),
    cl::cat(SearchCat));


cl::opt<bool> UseIterativeDeepeningTimeSearch(
    "use-iterative-deepening-time-search",
    cl::desc(
        "Use iterative deepening time search (experimental) (default=false)"),
    cl::init(false),
    cl::cat(SearchCat));

cl::opt<bool> UseBatchingSearch(
    "use-batching-search",
    cl::desc("Use batching searcher (keep running selected state for N "
             "instructions/time, see --batch-instructions and --batch-time) "
             "(default=false)"),
    cl::init(false),
    cl::cat(SearchCat));

cl::opt<unsigned> BatchInstructions(
    "batch-instructions",
    cl::desc("Number of instructions to batch when using "
             "--use-batching-search.  Set to 0 to disable (default=10000)"),
    cl::init(10000),
    cl::cat(SearchCat));

cl::opt<std::string> BatchTime(
    "batch-time",
    cl::desc("Amount of time to batch when using "
             "--use-batching-search.  Set to 0s to disable (default=5s)"),
    cl::init("5s"),
    cl::cat(SearchCat));

// add support for machine learning based search
cl::opt<std::string> ModelType(
    "model-type",
    cl::desc("Type of the machine learning model"),
    cl::init(""),
    cl::cat(SearchCat));

cl::opt<std::string> ModelPath(
     "model-path",
     cl::desc("Path of the machine learning model"),
     cl::init(""),
     cl::cat(SearchCat));

cl::opt<std::string> ScriptPath(
     "script-path",
     cl::desc("path that python script place"),
     cl::init(""),
     cl::cat(SearchCat));

cl::opt<std::string> PyPath(
     "py-path",
     cl::desc("path to corresponding python interpreter, default using system default interpreter"),
     cl::init(""),
     cl::cat(SearchCat));

cl::opt<bool> Sampling(
    "sampling",
    cl::desc("Whether to do sampling for ML searcher (default=false)"),
    cl::init(false),
    cl::cat(SearchCat));

} // namespace

void klee::initializeSearchOptions() {
  // default values
  if (CoreSearch.empty()) {
    if (UseMerge){
      CoreSearch.push_back(Searcher::NURS_CovNew);
      klee_warning("--use-merge enabled. Using NURS_CovNew as default searcher.");
    } else {
      CoreSearch.push_back(Searcher::RandomPath);
      CoreSearch.push_back(Searcher::NURS_CovNew);
    }
  }
}

bool klee::userSearcherRequiresMD2U() {
  return (std::find(CoreSearch.begin(), CoreSearch.end(), Searcher::NURS_MD2U) != CoreSearch.end() ||
          std::find(CoreSearch.begin(), CoreSearch.end(), Searcher::NURS_CovNew) != CoreSearch.end() ||
          std::find(CoreSearch.begin(), CoreSearch.end(), Searcher::NURS_ICnt) != CoreSearch.end() ||
          std::find(CoreSearch.begin(), CoreSearch.end(), Searcher::NURS_CPICnt) != CoreSearch.end() ||
          std::find(CoreSearch.begin(), CoreSearch.end(), Searcher::NURS_QC) != CoreSearch.end());
}


Searcher *getNewSearcher(Searcher::CoreSearchType type, RNG &rng, PTree &processTree, Executor &executor) {
  Searcher *searcher = nullptr;
  switch (type) {
    case Searcher::DFS: searcher = new DFSSearcher(); break;
    case Searcher::BFS: searcher = new BFSSearcher(); break;
    case Searcher::RandomState: searcher = new RandomSearcher(rng); break;
    case Searcher::RandomPath: searcher = new RandomPathSearcher(processTree, rng); break;
    case Searcher::NURS_CovNew: searcher = new WeightedRandomSearcher(WeightedRandomSearcher::CoveringNew, rng); break;
    case Searcher::NURS_MD2U: searcher = new WeightedRandomSearcher(WeightedRandomSearcher::MinDistToUncovered, rng); break;
    case Searcher::NURS_Depth: searcher = new WeightedRandomSearcher(WeightedRandomSearcher::Depth, rng); break;
    case Searcher::NURS_RP: searcher = new WeightedRandomSearcher(WeightedRandomSearcher::RP, rng); break;
    case Searcher::NURS_ICnt: searcher = new WeightedRandomSearcher(WeightedRandomSearcher::InstCount, rng); break;
    case Searcher::NURS_CPICnt: searcher = new WeightedRandomSearcher(WeightedRandomSearcher::CPInstCount, rng); break;
    case Searcher::NURS_QC: searcher = new WeightedRandomSearcher(WeightedRandomSearcher::QueryCost, rng); break;
    // support subpath guided search
    case Searcher::SGS_1: searcher = new SubpathGuidedSearcher(executor, 0); break;
    case Searcher::SGS_2: searcher = new SubpathGuidedSearcher(executor, 1); break;
    case Searcher::SGS_4: searcher = new SubpathGuidedSearcher(executor, 2); break;
    case Searcher::SGS_8: searcher = new SubpathGuidedSearcher(executor, 3); break;
    // add support for machine learning based search
    case Searcher::ML: searcher = new MLSearcher(executor, ModelType, ModelPath, ScriptPath, PyPath, Sampling); break;
  }

  return searcher;
}

Searcher *klee::constructUserSearcher(Executor &executor) {

  Searcher *searcher = getNewSearcher(CoreSearch[0], executor.theRNG, *executor.processTree, executor);

  if (CoreSearch.size() > 1) {
    std::vector<Searcher *> s;
    s.push_back(searcher);

    for (unsigned i = 1; i < CoreSearch.size(); i++)
      s.push_back(getNewSearcher(CoreSearch[i], executor.theRNG, *executor.processTree, executor));

    searcher = new InterleavedSearcher(s);
  }

  if (UseBatchingSearch) {
    searcher = new BatchingSearcher(searcher, time::Span(BatchTime),
                                    BatchInstructions);
  }

  if (UseIterativeDeepeningTimeSearch) {
    searcher = new IterativeDeepeningTimeSearcher(searcher);
  }

  if (UseMerge) {
    auto *ms = new MergingSearcher(searcher);
    executor.setMergingSearcher(ms);

    searcher = ms;
  }

  llvm::raw_ostream &os = executor.getHandler().getInfoStream();

  os << "BEGIN searcher description\n";
  searcher->printName(os);
  os << "END searcher description\n";

  return searcher;
}

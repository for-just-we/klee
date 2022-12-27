KLEE Symbolic Virtual Machine
=============================

[![Build Status](https://github.com/klee/klee/workflows/CI/badge.svg)](https://github.com/klee/klee/actions?query=workflow%3ACI)
[![Build Status](https://api.cirrus-ci.com/github/klee/klee.svg)](https://cirrus-ci.com/github/klee/klee)
[![Coverage](https://codecov.io/gh/klee/klee/branch/master/graph/badge.svg)](https://codecov.io/gh/klee/klee)

`KLEE` is a symbolic virtual machine built on top of the LLVM compiler
infrastructure. Currently, there are two primary components:

  1. The core symbolic virtual machine engine; this is responsible for
     executing LLVM bitcode modules with support for symbolic
     values. This is comprised of the code in lib/.

  2. A POSIX/Linux emulation layer oriented towards supporting uClibc,
     with additional support for making parts of the operating system
     environment symbolic.

Additionally, there is a simple library for replaying computed inputs
on native code (for closed programs). There is also a more complicated
infrastructure for replaying the inputs generated for the POSIX/Linux
emulation layer, which handles running native programs in an
environment that matches a computed test input, including setting up
files, pipes, environment variables, and passing command line
arguments.

For further information, see the [webpage](http://klee.github.io/).

# Add support for postconditioned symbolic execution

Content could refer to paper: [Eliminating Path Redundancy via
Postconditioned Symbolic Execution](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7835264&tag=1)

There is some difference between the implementation here and the algorithm described in original paper, the main motivation is reduce overhead:

- We abandon the concept of control location.

- Global variable $\Pi$(`Inst2PostCond`) here map `br` instruction to the its postcondition.


The pesudo code of the algo is:


```cpp
while (!states.empty()) {
    state = selectStates(); //DFS here
    // add constraint check, if satisfy already, terminate execution
    e = state.brs[-1];
    // if Inst2PostCond[e] must be true under state.constraints, then there are no need to further explore this state
    if (satisfy(state.constraints && ~Inst2PostCond[e])) {
        // normal symbolic execution process
    }
    else
        terminateEarly(state);
}
```

When taken a new state from state list, executor will first check whether the postcondition corresponding to this state has already cover the path of this state, if so, terminate early .Taken the example from the original paper:

```cpp
int main(){
    int a, b, c, res;
    klee_make_symbolic(&a,sizeof(a),"a");
    klee_make_symbolic(&b,sizeof(b),"b");
    klee_make_symbolic(&c,sizeof(c),"c");
    
    if (a <= 0)
        a = a + 10;
    else
        a = a - 10;
    
    if (a <= b)
        res = a - b;
    else
        res = a + b;
    
    if (res > c)
        res = 1;
    else
        res = 0;
    return 0;
}
```

The control-flow graph is:

```mermaid
flowchat
st=>start: Entry
e=>end: return 0;
cond1=>condition: a <= 0
op11=>operation: a = a + 10;
op12=>operation: a = a - 10;

cond2=>condition: a <= b
op21=>operation: res = a - b;
op22=>operation: res = a + b;

cond3=>condition: res > c
op31=>operation: res = 1
op32=>operation: res = 0


st->cond1
cond1(yes)->op11
cond1(no)->op12
op11->cond2
op12->cond2

cond2(yes)->op21
cond2(no)->op22
op21->cond3
op22->cond3

cond3(yes)->op31
cond3(no)->op32
op31->e
op32->e
```

Normally, there are 3 conditions (`c1: a<=0, c2: a<=b, c3: res>c`), when using DFS search

- When come across `a <= 0`, state will fork into state1 and state2.

- When state1 come across `a <= b`, state1 will fork into state11 and state12. Now state2 is on the bottom of state list.

- When state11 come across `res > c`, state11 will fork into state111 and state112. Now state12 and state2 are on the bottom of state list.

- When state111 and state112 terminates(1-3-5 and 1-3-6 in original paper). `Inst2PostCond = {c1: a<=0 & a+10<=b, c2: a+10<=b, c3:true}`

- When state12 start executing, executor will evaluate whether `state12.cond & ~Inst2PostCond[c2]` could satisfy. If so, continue this process until fork into state121 and state122.

- When state121 start executing, execute will evaluate whether `state121.cond & ~Inst2PostCond[c3]` could satisfy. Not, terminateEarly and update postconditions.

- state122 will be the same.




- `Expr::createIsZero` can be used to create `not` condition.

Cite

```
@article{2018Eliminating,
  title={Eliminating Path Redundancy via Postconditioned Symbolic Execution},
  author={ Yi, Q.  and  Yang, Z.  and  Guo, S.  and  Chao, W.  and  Chen, Z. },
  journal={IEEE Transactions on Software Engineering},
  volume={44},
  number={99},
  pages={25-43},
  year={2018},
}
```

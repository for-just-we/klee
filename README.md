KLEE Symbolic Virtual Machine
=============================

关于klee的信息可以参考[webpage](http://klee.github.io/)和[klee git](https://github.com/klee/klee)

这个repo基于klee-2.3 release改写，主要是将[learch](https://github.com/eth-sri/learch)的工作同步到最新2.3版本中，learch对klee的修改主要是添加了3个搜索策略:

- subpath guided search(SGS): 详情可以参考[paper](https://www.cs.ucdavis.edu/~su/publications/oopsla13-pgse.pdf)，对应 `SubpathGuidedSearcher` 类。

- learch machine learning search: learch提出的基于机器学习的搜索策略，对应 `MLSearcher` 类。

- get feature search: 对应 `GetFeaturesSearcher` 类，并不算是个严格意义上的搜索策略，主要目的是dump出每个state的feature信息。


# 修改信息

release 1增加了对subpath guided search的支持，learch的支持准备在下个release添加，相比原始klee-2.3修改包括:

- 在 `UserSearcher.cpp`，`Searcher.cpp`，`Searcher.h` 中添加了Subpath-guided search相关类，已经相关命令行参数

- 在 `ExecutionState` 类，`Executor` 类中添加相关成员变量支持（注意初始化）
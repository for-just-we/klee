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


相比2.3.1添加了对ML Search的支持，目前只支持给定pytorch model，MLSearcher会调用python解释器加载model进行state reward预测。
需要注意的是python代码必须和[model.py](https://github.com/eth-sri/learch/blob/master/learch/model.py) 保持一致。与2.3.1相比，klee命令行程序多出了以下选项

- `--feature-extract`: bool类型，是否提取特征，如果使用ml搜索策略必须开启

- `--search` 参数中多了 `ml` 选项，即机器学习搜索策略

- `--model-type=<value>`: 模型类型，有 `feedforward`, `linear`, `rnn` 3种选项

- `--model-path=<value>`: `<value>` 为pytorch模型保存路径

- `--script-path=<value>`: `<value>` 为python解释器路径，不设置的话会使用系统默认的

编译klee的时候需要添加参数 `-DPYTHON_INCLUDE_DIRS=<python_home>/include/pythonx.x -DLIB_PYTHON=<python_home>/lib/xx/libpythonx.x.so ..`，这里的python解释器路径应与 `--script-path` 中的保持一致

示例，假设我用anaconda3，根目录是 <conda_dir>，用到一个名为learch的虚拟环境，python版本3.8

- 编译的时候添加参数 `-DPYTHON_INCLUDE_DIRS=<conda_dir>/envs/learch/include/python3.8 -DLIB_PYTHON=<conda_dir>/envs/learch/lib/libpython3.8.so ..`，这一步cmake可能会报出警告，说 `<conda_dir>/envs/learch/lib/` 中存在的库可能会与 `/usr/lib` 冲突，不过问题不大，觉得不妥可以把 `libpython3.8.so` 复制到另一个空文件夹下。

- 运行的时候 `--script-path=<conda_dir>/envs/learch`。

# 目前存在的问题

`MLSearcher` 在当前版本运行存在一些问题，其它搜索策略（包括subpath）运行正常，原因在于 `Py_initialize` 会和klee posix runtime存在冲突，因此 `MLSearcher` 只能应用于不需要posix的示例。
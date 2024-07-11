# DEG:  Efficient Hybrid Vector Search Using the Dynamic Edge
Navigation Graph

## Introduction

The hybrid vector query, which calculates similarity scores for objects represented by two vectors using a weighted sum of their distances and employs a query-specific parameter $\alpha$ to determine the weight, has garnered significant attention from researchers recently due to the proliferation of bimodal data such as image-text pairs. Existing methods for hybrid vector queries face significant performance degradation with varying query $\alpha$ values. This is because existing Approximate Nearest Neighbors Search (ANNS) indexes assume static distances between objects. However, when $\alpha$ varies, the distances between objects change, causing the proximity properties of the ANNS indexes to become ineffective.

To address this, we introduce a novel Dynamic Relative Neighborhood Graph (D-RNG). The D-RNG maintains edges suitable for different $\alpha$ and assigns each edge an active range. During the search, it ignores certain edges based on active ranges and the query's $\alpha$. This strategy ensures that the graph maintains the property of the Relative Neighborhood Graph for varying $\alpha$, thereby ensuring high performance. To reduce indexing complexity, existing methods typically acquire an edge candidate set for each node and determine final edges from it. To acquire edge candidates suitable for varying $\alpha$, we propose considering the distances of the two vectors as two objective functions and using the Pareto frontiers of each node as the edge candidate set. To improve efficiency, we propose a novel greedy Pareto frontier search algorithm to find approximate Pareto frontiers. To reduce the search path length under varying $\alpha$, we further propose a new edge seed method. Using the techniques described above, we propose a new Dynamic Edge Navigation Graph (\method) to approximate the D-RNG. Extensive experiments on real-world datasets show that \method demonstrates superior performance compared to existing methods under varying query $\alpha$ values.


This project contains the code,, optimal parameters, and other detailed information used for the experiments of our paper. It is worth noting that we reimplement all algorithms based on exactly the same design pattern, programming language and tricks, and experimental setup, which makes the comparison more fair. 


## Datasets

Our experiment involves four real-world datasets. Note that, all base data and query data are converted to `fvecs` format, and ground-truth data is converted to `ivecs` format. Please refer [here](http://yael.gforge.inria.fr/file_format.html) for the description of `fvecs` and `ivecs` format.

## Parameters

For the  parameters of each algorithm on all experimental datasets, see the code.

## Usage

### Prerequisites

* GCC 4.9+ with OpenMP
* CMake 2.8+
* Boost 1.55+

### Compile on Linux

```shell
$ mkdir build && cd build/
$ cmake ..
$ make -j
```

### Index construction evaluation

Then, you can run the following instructions for build graph index.

```shell
cd ./build/test/
./main algorithm_name dataset_name \alpha max_distance_1 max_distance_2 build
```


### Search performance
With the index built, you can run the following commands to perform the search. Related information about the search such as search time, distance evaluation times, candidate set size, average query path length, memory load can be obtained or calculated according to the output log information.

```shell
cd ./build/test/
./main algorithm_name dataset_name \alpha max_distance_1 max_distance_2 search
```


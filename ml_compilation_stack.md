Source: [apxml - ML Compiler Optimizations](https://apxml.com/courses/intro-ml-compiler-optimization)

## Glossary 
 - Chapter 1: ML Compilation Stack - *AOT v JIT, Tracing and graph capture*
 - Chapter 2: Intermediate Representatoins - *data flow, tensor shapes and dynamic shapes*
 - Chapter 3: Graph-Level Optimizations - *op fusion, dce, cse*
 - Chapter 4: Kernel and Loop Optmizatoins - *tiling, parallelization, latency hiding*
 - Chapter 5: Auto-tuning and Code-Gen - *search space, cost models*


### Chapter 1 

In machine learning we're describing mathematical intent rather than explicit machine intructions
ML compilers - focus on tensor algebra and massive parallelism  as opposed to regular compilers which optimise for scalar logic. 

#### Framework-Hardware Gap


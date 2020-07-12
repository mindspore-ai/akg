
This is a revised apache/incubator-tvm 0.6 for Auto Kernel Generator.
What we are using:
* TVM IR
* Arith Simplify
* Several passes
* DSL api
* topi

Compared with original apache/incubator-tvm 0.6, we have modified some codes:

1. Changed namespace from tvm to air.
2. For these modules(listed above) that we used, we have added or deleted some codes targeted to Auto Kernel Generator. 
   For detailed changes, please refer to the source codes, where we have listed the modification at the head of file.


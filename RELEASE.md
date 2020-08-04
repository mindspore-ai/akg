# Release 0.6.0-beta
## Major Features and Imporvements
* AutoPoly refactor to support integrating multi-backend targets easily
  * Employ a pass/passmgr framework to manage all the transformations of ISL schedule tree in which transformation such as InitialSchTree and tileOuterBand would be considered as a pass to schedule tree.
  * Refactor some data structure of poly so that they can de-couple with Davinci chips.
* Backend refactoring
  * Enhance min alignment analysis with more accurate propagate conditions.
  * Finetune pragma using alignment information before EmitInsn pass.
  * Simplify EmitInsn pass by unifying the emit method for different patterns.
* Change the way of using TVM
  * Delete the repository ktvm and reference TVM directly in sourcecode(third_party/incubator-tvm).
  * Enable GPU operators generation which was tailored in ktvm.

## Bugfixes
* fix wrong hoist problem in multicore loop switch hoist pass(!87).
* fix scalar rearrange bug(!84).
* fix matmul tuning and support all space tuning(!73).
* fix variable broadcast_idx redefinition error when pragma dma_copy is replaced by opt_broadcast(!45).
* fix the bug in broadcast_rewrite(!22).
* fix bugs of multi-core processing(!33).
* fix a bug that extra pipe_barrier inserted in the loop(!30).
* fix inefficient auto tiling for axis with tail and remove duplicated check(!6).

## Contributors
Thanks goes to these wonderful people:
brovensmile, chengyun, chenlei_autodiff, chengbin, ConnZhai, fuxiaoteng, gaoxiong, gengzhen, hanhuifeng, KasonChan, luoyin, lvwenyuan, peiwenfang, xuhui, yangsijia, wangzhuo325, wYann

Contributions of any kind are welcome!

# Release 0.5.0-beta
## Major Features
* Support auto-schedule and code-generation on Ascend platform.
* Provide C++ APIs of basic operators used in MindSpore.
* Support Elementwise-Elementwise, Reduce-Elementwise fusion patterns in Bert.
* Support LambUpdateWithLR, LambNextMv, BatchMatmul optimazition for Bert.

## Initial Version
* Upload the initial framework
* Basic support for Ascend910 platform
* Integration with GraphKernel

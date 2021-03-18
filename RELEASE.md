# Release 1.1.1
## Major Features and Improvements
* Enable Tensor core when processing GEMM operators in AKG by using poly to create the schedule needed by tensor core pass automatically;
* Implemented an akg mma lib with inlined ptx codes instead of wmma interface of cuda;
* Enable one-dimensional mapping to optimize memory promotion.

## Bugfixes
* Fix Segmentation fault in Mapping OuterBand in mindspore (!321).
* Fix bugs for memory promotion issues  (!306).
* Fix bugs during gen tuning space for scalar ops (!326).

## Contributors
Thanks goes to these wonderful people:

chengyun, chendeshi, chenlei_autodiff, gengzhen, hanhuifeng, lvwenyuan, lishanni513, hujiahui8, polyhedral, shiliang,  wYann, xixixian, xxxxxxw, xuhui, xiaruijie, yangsijia, yiyanzhi, zhangzhaochuang, zhengzuohe

Contributions of any kind are welcome!

# Release 1.1.0
## Major Features and Improvements
* GPU operators improvements
  * Propose a new strategy to handle the reduction operators: The reduce axises would be detected and rescheduled as a seperated band in the schedule tree and then mapping to blocks, then it will call the akg_reduce_lib which using atomic operation to do reduction in the codegen pass. The experimental results show that AKG improves the execution performance relative to cudnn in the large shape cases;
  * Optimize the auto-tiling algorithms which can improve the performance of reduction operators dramatically in most scenarios.
* Support AutoTuning for composite operators on GPU;
* Refactor composite framework to enable optimization in DSL level;
* Enhance CSE to support eliminating redundant vmadd on Ascend;
* Update scipy to 1.5.3.

## Bugfixes
* TensorAdd support FRACTAL_NZ and DefaultFormat(!228).
* GPU fix cast: fp32 -> uint8(!216).
* bugfix: Fix bug in opt_broadcast(!272).
* fix vadds for int32(!250).

## Contributors
Thanks goes to these wonderful people:

chengyun, chendeshi, chenlei_autodiff, gaoxiong, gengzhen, guanxiaowei, hanhuifeng, laekov, luoyin, lvwenyuan, liuchang, lishanni513, lingyunli63, polyhedral, shiliang, wYann, wangrao124, xiaruijie, xixixian, xuhui, 要术甲杰, yiyanzhi_akane, yangshuo, yangsijia, zhangzhaochuang, zhengzuohe, zhangrenwei, zengzitao

Contributions of any kind are welcome!

# Release 1.0.0
## Major Features and Improvements
* GPU Support
  * AKG now can generate gpu cuda kernel with no-schedule by using polyhedral techniques, which will create initial schedule, tile outerBands, map with blocks and threads and memory promotion automatically in the AutoPoly pass.
  * Some primitive and fused operators(most are element-wise operators and reduce operators) were added, as well as corresponding testcases.
* Schedule-templates enhancement
  * Optimize the TVM original schedule-templates to get better performance in some reduce cases.
  * Support fusing multi-outputs into one kernel for element-wise operators.
* Davinci Enhancement
  * Eliminate unnecessary broadcast by transforming the element-wise computation, such as `D[i, j] = A[i] + B[i, j] + C[i]` -> `D[i, j] = A[i] + C[i] + B[i, j]`, which satisfies commutative law and associative law.
  * Enhance the pass to_three_address to match more cases for vmadd.

## Bugfixes
* fix a bug that random test case segment_max failed(!127).
* fix the permisson denied error of rewriting meta_file with same name(!147).
* fix warning for unsupported gpu built-in ops(!148).

## Contributors
Thanks goes to these wonderful people:

baita, ConnZhai, gengzhen, guanxiaowei, hanhuifeng, hujiahui8, laekov, lvwenyuan, lishanni513, lingyunli63, polyhedral, wYann, wangrao124, xixixian, xuhui, 要术甲杰, yiyanzhi_akane, yangsijia, zhengzuohe, zhangrenwei, zengzitao

Contributions of any kind are welcome!

# Release 0.7.0-beta
## Major Features and Improvements
* Backend refactoring
  * Rewrite instruction args calculation module in EmitInsn by implementing a new computing strategy based on axis spliting, which achieved improvement both on performance and code simplicity.

## Bugfixes
* fix dump code error when running gpu operators and set env MS_AKG_DUMP_CODE=ON(!113).

## Contributors
Thanks goes to these wonderful people:

lvwenyuan, shiliang, xuhui, wYann

Contributions of any kind are welcome!

# Release 0.6.0-beta
## Major Features and Improvements
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

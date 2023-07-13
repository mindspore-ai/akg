# MindAKG Release Notes

[查看中文](./RELEASE_CN.md)

## MindAKG 2.1.0 Release Notes

### Bug Fixes
* Fixed the inconsistencies issue during build to support PolyTops on Ascend. [!908](https://gitee.com/mindspore/akg/pulls/908) (Ascend)
* Fixed the adaptability issue to CANN 6.4. [!911](https://gitee.com/mindspore/akg/pulls/911) (Ascend)
* Fixed the adaptability issue to MatMul on MSLite Ascend. [!914](https://gitee.com/mindspore/akg/pulls/914) (Ascend)

### Contributors

Thanks goes to these wonderful people:

liuchao, huoxinyou, yangsijia, Harenome Ranaivoarivony-Razanajato

## MindAKG 2.0.0 Release Notes

### Major Features and Improvements

### Bug Fixes

* Fixed the dependency issue on PolyTOPS during building. [!873](https://gitee.com/mindspore/akg/pulls/873) (All)
* Fixed the data type issue of int64 in GPU code generation. [!868](https://gitee.com/mindspore/akg/pulls/868) (GPU)

### Contributors

Thanks goes to these wonderful people:

liuchao, yiyanzhi, hujiahui8, xuhui, yangshuo, nelson.lossing

## MindAKG 1.9.0 Release Notes
### Major Features and Improvements

* Support the auto-generation of cpu-pool2d(avg/max/global). [!763](https://gitee.com/mindspore/akg/pulls/763) (CPU)
* Integerate Symbolic Tiling algorithm for AKG. [!764](https://gitee.com/mindspore/akg/pulls/764) (Ascend)
* AutoTiling optimization on GPU Backend. [!784](https://gitee.com/mindspore/akg/pulls/784) [!785](https://gitee.com/mindspore/akg/pulls/785) [!783](https://gitee.com/mindspore/akg/pulls/783)  (GPU)
* optimization for Transpose Operator. [!811](https://gitee.com/mindspore/akg/pulls/811) (CPU)

### Bug fixes
* Fix the bug that analysis error of read-write relationship in multi-filter.[!778](https://gitee.com/mindspore/akg/pulls/778) (All)
* Fix the bug when batch_matmul recognizes the axis. [!807](https://gitee.com/mindspore/akg/pulls/807) (CPU)
* Fix the bug when gcc version is less then 7.3. [!817](https://gitee.com/mindspore/akg/pulls/817) (All)

 ### Contributors

Thanks goes to these wonderful people:

yangsijia, polyhedral, zhangrenwei, yiyanzhi, hujiahui8, zhengzuohe, zhangzhaochuang, xuhui, liuchao, xiaruijie, DeshiChen, nelson.lossing, zhenzhang, chenlei_autodiff,  wYann, hanhuifeng, gaoxiong, zichun_ye, chengbin, wangrao124, huangmengxi, harenome, huoxinyou, yanglinfeng, Etienne

## MindAKG 1.8.0 Release Notes

### Major Features and Improvements

* Integrate the CPU compiling process for CustomOp. [!612](https://gitee.com/mindspore/akg/pulls/612) (CPU)
* Optimize high-dimensional CSR OP. [!664](https://gitee.com/mindspore/akg/pulls/664) (GPU)
* Update the restart process of Ascend backend. [!701](https://gitee.com/mindspore/akg/pulls/701) (Ascend)
* Support the auto-generation of CSRMM operator. [!709](https://gitee.com/mindspore/akg/pulls/709) (GPU)
* Integrate usage of intrinsic directives for the new Scheduler. [!715](https://gitee.com/mindspore/akg/pulls/715) (All backends)

### Bug fixes

* Fix the bug of reshape elimination for the optimization process of fused operators compiling. [!707](https://gitee.com/mindspore/akg/pulls/707) (All)
* Fix the bug of to_three_address pass. [!728](https://gitee.com/mindspore/akg/pulls/728) (Ascend)
* Fix the bug of Operator types matching rules. [!730](https://gitee.com/mindspore/akg/pulls/730) (All)

### Contributors

Thanks goes to these wonderful people:

yangsijia, polyhedral, zhangrenwei, yiyanzhi, hujiahui8, zhengzuohe, zhangzhaochuang, xuhui, liuchao, xiaruijie, DeshiChen, nelson.lossing, zhenzhang,

chenlei_autodiff,  wYann, hanhuifeng, gaoxiong, zichun_ye, chengbin, wangrao124, huangmengxi, harenome, huoxinyou, yanglinfeng, Etienne

## MindAKG 1.7.0 Release Notes

### Major Features and Improvements

* Support a new loop intrinsic for MindSpore HybridDSL. [!560](https://gitee.com/mindspore/akg/pulls/560) (ALL backends)
* Update the shared/local promotion strategy on GPU backend. [!556](https://gitee.com/mindspore/akg/pulls/556) (GPU)
* Use the new interface provided by isl to reconstruct the reschedule pass after the pluto scheduling algorithm on the Ascend backend.[!512](https://gitee.com/mindspore/akg/pulls/512) (Ascend)

### Bug fixes

* Fix the bug of Gemm/convolution operator due to repeated tiling. [!582](https://gitee.com/mindspore/akg/pulls/582) (GPU)
* Fix the bug in the reduce operator when the Y direction is reduced and X is small. [!559](https://gitee.com/mindspore/akg/pulls/559) (CPU)

### Contributors

Thanks goes to these wonderful people:

yangsijia, polyhedral, zhangrenwei, yiyanzhi, hujiahui8, zhengzuohe, zhangzhaochuang, xuhui, liuchao, xiaruijie, DeshiChen, nelson.lossing, zhenzhang,

chenlei_autodiff, lingyunli63, wYann, hanhuifeng, gaoxiong, zichun_ye, chengbin, wangrao124, huangmengxi, harenome, lear, huoxinyou, yanglinfeng, Etienne, Trump

## Release 1.6.0

### Major Features and Improvements

* [STABLE] CPU backend support: AKG now supports automatic generation of high-performance code for CPUs with different architectures. [!413](https://gitee.com/mindspore/akg/pulls/413) (CPU)
* [STABLE] CSR operators support: Develops csr operators(csrmv/csr mul/csr reduce_sum)，and provides optimizing strategy to handle csr operators with dynamic upper bound. [!407](https://gitee.com/mindspore/akg/pulls/407) (GPU)
* [STABLE] Scheduler Optimizing: Outermost band node now will be treated as multiple filter nodes if there is no dependency relationship between multiple statements. [!460](https://gitee.com/mindspore/akg/pulls/460) (GPU)
* [STABLE] Replace gmp with imath on isl.[!455](https://gitee.com/mindspore/akg/pulls/455) (ALL)
* [STABLE] Apply Autodiff in Custom Op. [!464](https://gitee.com/mindspore/akg/pulls/464) (ALL)

### Bug fixes

* fix bugs in cpu tuning process:add "need_warm_up" attrs to avoid too many warm-up on cpu profiling.[!495](https://gitee.com/mindspore/akg/pulls/495) (CPU)
* fix auto-tiling output mem-flow bug:modify the GEMM's auto-tiling strategy to match the actual target constraint.[!504](https://gitee.com/mindspore/akg/pulls/504) (ASCEND)
* fix muilt output bug in emitter: when emit multi-output, should not convert real inputs.[!506](https://gitee.com/mindspore/akg/pulls/332) (ALL)

### Contributors

Thanks goes to these wonderful people:

yangsijia, polyhedral, zhangrenwei, yiyanzhi, hujiahui8, zhengzuohe, zhangzhaochuang, xuhui, liuchao, xiaruijie, DeshiChen, nelson.lossing, zhenzhang, chenlei_autodiff, lingyunli63, wYann, hanhuifeng, gaoxiong, zichun_ye, chengbin, wangrao124, huangmengxi, harenome, lear, huoxinyou, yanglinfeng, Etienne, Trump

Contributions of any kind are welcome!

## Release 1.5.0

### Major Features and Improvements

* [STABLE] New operators developing: Tensor of Tensor operators(Gather/GatherNd/TensorScatterAdd/UnsortedSegmentSum), which can be used to support GNN networks.[!323](https://gitee.com/mindspore/akg/pulls/323)(GPU)
* [STABLE] Add a topi op of UserDefine Op in AKG, which can be compiled from func_source_string or op_imply_path.[!319](https://gitee.com/mindspore/akg/pulls/319)(GPU)
* [STABLE] Lower interface Refactor: add StageLower for stage lower case.[!310](https://gitee.com/mindspore/akg/pulls/310)(GPU)
* [STABLE] The profiling suit for new runtime process.[!306](https://gitee.com/mindspore/akg/pulls/306)(ASCEND)

### Bug fixes

* Fixed memory promotion bug: sort the clusters before merging.[!338](https://gitee.com/mindspore/akg/pulls/338) (ASCEND)
* Fixed irregular-reduce bug: replace shfl.down with shared memory reducetion.[!332](https://gitee.com/mindspore/akg/pulls/332) (GPU)
* Fixed foldDimension bug: build wrong axis relation of relation.[!302](https://gitee.com/mindspore/akg/pulls/302) (GPU)

### Contributors

Thanks goes to these wonderful people:

yangsijia, xxxxxxw, polyhedral, zhangrenwei, yiyanzhi, xixixian, hujiahui8, zhengzuohe, lishanni, zhangzhaochuang, xuhui, liu
chao, gengzhen, xiaruijie,chenlei_autodiff, lingyunli63, wYann, lvwenyuan, peiwenfang, hanhuifeng, gaoxiong, chengyun
Contributions of any kind are welcome!

## Release 1.3.0

### Major Features and Improvements

* [STABLE] Support optimizing GEMM && Conv by using polyhedral + Tensorcore, as well as providing an akg::fragment_add/sub/mul/div library for GEMM op fusions. [!156](https://gitee.com/mindspore/akg/pulls/156) (GPU)
* [STABLE] Optimize layout related operators(Transpose && pad/unpad) by adjusting the autotiling strategy and solving bank conflict for these ops. [!152](https://gitee.com/mindspore/akg/pulls/152/) (GPU)
* [STABLE] Add Kahan algorithm for reducetion operators. [!107](https://gitee.com/mindspore/akg/pulls/107) (GPU)
* [STABLE] Support transdata + matmul prefusion pattern in akg. [!103](https://gitee.com/mindspore/akg/pulls/103) (ASCEND)

### Bug fixes

* Fixed stitch_fusion bug when store is shared but load is not shared. [!109](https://gitee.com/mindspore/akg/pulls/109) (GPU)
* Fixed reshape bug: when add reshape, should update tensor shape. [!140](https://gitee.com/mindspore/akg/pulls/140) (GPU)
* Fixed autofuse bug: when set autofuse config, should record broadcast. [!111](https://gitee.com/mindspore/akg/pulls/111) (GPU)

### Contributors

Thanks goes to these wonderful people:

yangsijia, xxxxxxw, polyhedral, zhangrenwei, yiyanzhi, xixixian, hujiahui8, zhengzuohe, lishanni, zhangzhaochuang, xuhui, liu
chao, gengzhen, xiaruijie,chenlei_autodiff, lingyunli63, wYann, lvwenyuan, peiwenfang, hanhuifeng, gaoxiong, chengyun
Contributions of any kind are welcome!

## Release 1.2.0

## Major Features and Improvements

* [STABLE] Rebuild the AKG repository for providing a new way to support ascend backend by linking a static library contained all the ascend passes. (Ascend)
* [STABLE] Optimize the reduction add operation in ascend backend. (Ascend)
* [STABLE] Add support for tuning elemwise&&reduction operators. (GPU)

### Bug fixes

* Fixed a problem that data prefetch cannot be enabled by attributes in DSL.
* Fixed bugs of autotiling algorithms (tiling too small, cannot adapted matmul+bias, etc.) in Ascend platform.
* Fixed local memory promotion for large thread (2980!)
* Fixed reduce binding dimension issue on gpu platform (ff38!)

### Contributors

Thanks goes to these wonderful people:

yangsijia, xxxxxxw, polyhedral, zhangrenwei, yiyanzhi, xixixian, hujiahui8, zhengzuohe, lishanni, zhangzhaochuang, xuhui, liuchao, gengzhen, xiaruijie,
chenlei_autodiff, lingyunli63, wYann, lvwenyuan, peiwenfang, hanhuifeng, gaoxiong, chengyun
Contributions of any kind are welcome!

## Initial Version

* Upload the initial framework
* Basic support for Ascend910 platform and gpu v100
* Integration with GraphKernel fusion of MindSpore.

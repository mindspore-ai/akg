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


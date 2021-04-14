## Release 1.2.0
### Bug fixes
  * Fixed local memory promotion for large thread (2980!)
  * Fixed reduce binding dimension issue on gpu platform (ff38!)

## Release 1.2.0-rc1
### Major Features and Improvements
  * [STABLE] Rebuild the AKG repository for providing a new way to support ascend backend by linking a static library contained all the ascend passes. (Ascend)
  * [STABLE] Optimize the reduction add operation in ascend backend. (Ascend)
  * [STABLE] Add support for tuning elemwise&&reduction operators. (GPU)

### Bug fixes
  * Fixed a problem that data prefetch cannot be enabled by attributes in DSL.
  * Fixed bugs of autotiling algorithms (tiling too small, cannot adapted matmul+bias, etc.) in Ascend platform.

### Contributors
Thanks goes to these wonderful people:

yangsijia, xxxxxxw, polyhedral, zhangrenwei, yiyanzhi, xixixian, hujiahui8, zhengzuohe, lishanni, zhangzhaochuang, xuhui, liuchao, gengzhen, xiaruijie, 
chenlei_autodiff, lingyunli63, wYann, lvwenyuan, peiwenfang, hanhuifeng, gaoxiong, chengyun
Contributions of any kind are welcome!

## Initial Version
* Upload the initial framework
* Basic support for Ascend910 platform and gpu v100
* Integration with GraphKernel fusion of MindSpore.


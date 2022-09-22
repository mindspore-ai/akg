# MindAKG Release Notes

[View English](./RELEASE.md)

## MindAKG 1.9.0 Release Notes

### 主要特性和增强

* 支持CPU后端pool2d相关算子(avg/max/global)的自动生成流程。[!763](https://gitee.com/mindspore/akg/pulls/763) (CPU)
* AKG集成符号切分算法。[!764](https://gitee.com/mindspore/akg/pulls/764) (Ascend)
* GPU后端自动切分算法优化。 [!784](https://gitee.com/mindspore/akg/pulls/784) [!785](https://gitee.com/mindspore/akg/pulls/785) [!783](https://gitee.com/mindspore/akg/pulls/783)  (GPU)
* CPU后端Transpose算子性能优化。[!811](https://gitee.com/mindspore/akg/pulls/811) (CPU)

### Bug 修复

* 修复了多filter场景中读写依赖分析时的bug。[!778](https://gitee.com/mindspore/akg/pulls/778) (All)
* 修复了CPU后端batchMatmul识别轴流程中的bug。[!807](https://gitee.com/mindspore/akg/pulls/807) (CPU)
* 修复了系统环境gcc版本低于7.3时，编译reduce算子时存在的bug。[!817](https://gitee.com/mindspore/akg/pulls/817) (All)

### 贡献者

感谢以下开发者做出的贡献：

yangsijia, polyhedral, zhangrenwei, yiyanzhi, hujiahui8, zhengzuohe, zhangzhaochuang, xuhui, liuchao, xiaruijie, DeshiChen, nelson.lossing, zhenzhang, chenlei_autodiff, wYann, hanhuifeng, gaoxiong, zichun_ye, chengbin, wangrao124, huangmengxi, harenome, huoxinyou, yanglinfeng, Etienne


## MindAKG 1.8.0 Release Notes

### 主要特性和增强

* 自定义算子支持CPU编译流程。[!612](https://gitee.com/mindspore/akg/pulls/612) (CPU)
* 支持高维CSR算子及其性能优化。[!664](https://gitee.com/mindspore/akg/pulls/664) (GPU)
* 更新昇腾后端Poly重启动流程策略。[!701](https://gitee.com/mindspore/akg/pulls/701) (Ascend)
* 支持CSRMM算子的自动生成。[!709](https://gitee.com/mindspore/akg/pulls/709) (GPU)
* 新调度器支持调度原语辅助调度。[!715](https://gitee.com/mindspore/akg/pulls/715) (All backends)

### Bug 修复

* 修复了融合算子编译优化过程中reshape消除引入的bug。[!707](https://gitee.com/mindspore/akg/pulls/707) (All)
* 修复了三地址pass的一个bug。[!728](https://gitee.com/mindspore/akg/pulls/728) (Ascend)
* 修复了算子类别识别流程中的bug。[!730](https://gitee.com/mindspore/akg/pulls/730) (All)

### 贡献者

感谢以下开发者做出的贡献：

yangsijia, polyhedral, zhangrenwei, yiyanzhi, hujiahui8, zhengzuohe, zhangzhaochuang, xuhui, liuchao, xiaruijie, DeshiChen, nelson.lossing, zhenzhang,

chenlei_autodiff, wYann, hanhuifeng, gaoxiong, zichun_ye, chengbin, wangrao124, huangmengxi, harenome, huoxinyou, yanglinfeng, Etienne

## MindAKG 1.7.0 Release Notes

### 主要特性和增强

* 为MindSpore HybridDSL特性提供循环原语支持。[!560](https://gitee.com/mindspore/akg/pulls/560) (所有后端)
* 更新GPU后端的shared/local内存提升策略。[!556](https://gitee.com/mindspore/akg/pulls/556) (GPU)
* 昇腾后端使用isl提供的新接口重构了pluto调度算法后的reschedule pass。[!512](https://gitee.com/mindspore/akg/pulls/512) (Ascend)

### Bug修复

* 修复了Gemm/卷积算子中重复切分的bug。 [!582](https://gitee.com/mindspore/akg/pulls/582) (GPU)
* 修复了reduce算子中，对Y方向规约且X较小时的bug。[!559](https://gitee.com/mindspore/akg/pulls/559) (CPU)

### 贡献者

感谢以下开发者做出的贡献:

yangsijia, polyhedral, zhangrenwei, yiyanzhi, hujiahui8, zhengzuohe, zhangzhaochuang, xuhui, liuchao, xiaruijie, DeshiChen, nelson.lossing, zhenzhang,

chenlei_autodiff, lingyunli63, wYann, hanhuifeng, gaoxiong, zichun_ye, chengbin, wangrao124, huangmengxi, harenome, lear, huoxinyou, yanglinfeng, Etienne, Trump

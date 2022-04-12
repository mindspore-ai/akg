## Release 1.7.0

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

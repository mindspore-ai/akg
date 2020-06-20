/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DNN_BASE_HPP__
#define DNN_BASE_HPP__

#include "cce/cce_def.hpp"

namespace cce {

/**
 * @ingroup dnn
 * @brief tiling para
 */
typedef struct tagCcWeightCompressInfo {
  uint32_t blockRow;     /**< block row */
  uint32_t blockCol;     /**< block col */
  uint32_t fractalK;     /**< fractal K */
  uint32_t fractalN;     /**< fractal N */
  uint32_t lastFractalK; /**< K of last fractal */
  uint32_t lastFractalN; /**< N of last fractal */
  uint32_t cubeSize;     /**< cube's length */
  uint32_t loadDir;      /**< data load directtiono 0??col load     1:row load*/
} ccWeightCompressInfo_t;

/**
 * @ingroup dnn
 * @brief compress table info
 */
typedef struct tagCcWeightCompressTab {
  uint16_t dataLen : 14;  /**< 0: data length in 128 Byte */
  uint16_t storeFlag : 1; /**< 0: compressed addr = original addr, 1: compressed addr = original addr + 256 Byte */
  uint16_t dataType : 1;  /**< 0: original data, 1: compressed data */
} ccWeightCompressTab_t;

/**
 * @conv quantize dnn vector mode/scalar mode
 */
typedef enum {
  QUANT_ALGO_NON_OFFSET = 0,
  QUANT_ALGO_HALF_OFFSET = 1,
  QUANT_ALGO_ALL_OFFSET = 2,
  QUANT_ALGO_BUTT
} ccQuantizeAlgo_t;
typedef enum { SCALE_VEC = 0, SCALE_SCALAR = 1, SCALE_TYPE_BUTT } ccConvolutionScaleType_t, ccScaleType_t;

/**
 * @conv quantize dnn sqrt mode/non sqrt mode
 */
typedef enum {
  SCALE_NORMAL = 0,
  SCALE_SQRT = 1,
  SCALE_VALUE_MODE_BUTT
} ccConvolutionScaleValueMode_t,
    ccScaleValueMode_t;

typedef struct {
  float scaleW;
  float scaleD;
  float scaleDNext;
  uint8_t offsetW;
  uint8_t offsetD;
  uint8_t offsetDNext;
} ccQuantAllOffsetPara_t;

typedef struct tagCcVecQuantizePara {
  float scale;
  uint16_t offset;
  uint16_t rrv;  // 32byte align
} ccVecQuantizePara_t;

/**
 * @ingroup dnn
 * @brief format of tensor
 */
typedef enum tagCcTensorFormat {
  CC_TENSOR_NCHW = 0,  /**< NCHW */
  CC_TENSOR_NHWC,      /**< NHWC */
  CC_TENSOR_ND,        /**< Nd Tensor */
  CC_TENSOR_NC1HWC0,   /**< NC1HWC0 */
  CC_TENSOR_FRACTAL_Z, /**< FRACTAL_Z */
  CC_TENSOR_NC1C0HWPAD,
  CC_TENSOR_NHWC1C0,
  CC_TENSOR_FSR_NCHW,
  CC_TENSOR_FRACTAL_DECONV,
  CC_TENSOR_C1HWNC0,
  CC_TENSOR_FRACTAL_DECONV_TRANSPOSE,
  CC_TENSOR_FRACTAL_DECONV_SP_STRIDE_TRANS,
  CC_TENSOR_NC1HWC0_C04,   /**< NC1HWC0, C0 =4*/
  CC_TENSOR_FRACTAL_Z_C04, /**< FRACZ?????C0 =4 */
  CC_TENSOR_CHWN,
  CC_TENSOR_FRACTAL_DECONV_SP_STRIDE8_TRANS,
  CC_TENSOR_HWCN,
  CC_TENSOR_NC1KHKWHWC0, /** < KH,KW kernel h& kernel w maxpooling max output format*/
  CC_TENSOR_HASHTABLE_LOOKUP_LOOKUPS = 20,
  CC_TENSOR_HASHTABLE_LOOKUP_KEYS,
  CC_TENSOR_HASHTABLE_LOOKUP_VALUE,
  CC_TENSOR_HASHTABLE_LOOKUP_OUTPUT,
  CC_TENSOR_HASHTABLE_LOOKUP_HITS = 24,
  CC_TENSOR_C1HWNCoC0, /**< C1,H,W,N,Co,C0 6D diagonal format*/
  CC_TENSOR_RESERVED
} ccTensorFormat_t;

/**
 * @ingroup dnn
 * @brief format of compare
 */
typedef enum tagCcCompareType {
  CC_COMPARE_TYPE_LESS = 0,
  CC_COMPARE_TYPE_LESS_EQUAL,
  CC_COMPARE_TYPE_NOT_EQUAL,
  CC_COMPARE_TYPE_EQUAL,
  CC_COMPARE_TYPE_GREATER,
  CC_COMPARE_TYPE_GREATER_EQUAL,
  CC_COMPARE_TYPE_RESERVED
} ccCompareType_t;

/**
 * @ingroup dnn
 * @brief propagate Nan
 */
typedef enum tagCcNanPropagation {
  CC_NAN_NOT_PROPAGATE = 0, /**< Nan numbers are not propagated */
  CC_NAN_PROPAGATE,         /**< Nan numbers are propagated */
  CC_NAN_PROPAGATE_RESERVED
} ccNanPropagation_t;

/**
 * @ingroup dnn
 * @brief algorithm of convolution forward
 */
typedef enum tagCcConvolutionFwdAlgo {
  CC_CONVOLUTION_FWD_ALGO_GEMM = 0, /**< matrix gemm algo */
  CC_CONVOLUTION_FWD_ALGO_WINOGRAD, /**< Winograd Transform algo */
  CC_CONVOLUTION_FWD_ALGO_GEMM_ACCU_FLOAT32,
  CC_CONVOLUTION_FWD_ALGO_RESERVED
} ccConvolutionFwdAlgo_t;

#define ccCorrelationFwdAlgo_t ccConvolutionFwdAlgo_t

typedef enum tagCcConvolutionBwdAlgo {
  CC_CONVOLUTION_BWD_ALGO_GEMM = 0, /**< matrix gemm algo */
  CC_CONVOLUTION_BWD_ALGO_WINOGRAD, /**< Winograd Transform algo */
  CC_CONVOLUTION_BWD_ALGO_GEMM_CO2IMG,
  CC_CONVOLUTION_BWD_FILTER_GEM_ALGO,
  CC_CONVOLUTION_BWD_ALGO_RESERVED
} ccConvolutionBwdAlgo_t;

#define ccCorrelationBwdAlgo_t ccConvolutionBwdAlgo_t

/**
 * @ingroup dnn
 * @brief algorithm of FullConnect forward
 */
typedef enum tagCcFullConnectFwdAlgo {
  CC_FULLCONNECT_FWD_ALGO_HALF = 0,
  CC_FULLCONNECT_FWD_ALGO_FLOAT32
} ccFullConnectFwdAlgo_t;

/**
 * @ingroup dnn
 * @brief mode of convolution
 */
typedef enum tagCcConvolutionMode {
  CC_CONV_CONVOLUTION = 0,   /**< math convolution */
  CC_CONV_CROSS_CORRELATION, /**< cross-correlation convolution */
  CC_CONV_DECONVOLUTION,     /**< deconvolution, also named transposed convolution*/
  CC_CONV_MODE_DEPTHWISE,    /**< depthwise convolution*/
  CC_CONV_MODE_RESERVED
} ccConvolutionMode_t;

#define ccCorrelationMode_t ccConvolutionMode_t

/**
 * @ingroup dnn
 * @brief mode of pooling
 */
typedef enum tagCcPoolingMode {
  CC_POOLING_MAX = 0,  /**< max pooling */
  CC_POOLING_AVG,      /**< average pooling */
  CC_POOLING_L2,       /**< L2 pooling */
  CC_POOLING_AVG_FP32, /**< average pooling for training */
  CC_POOLING_RESERVED
} ccPoolingMode_t;

/**
 * @ingroup dnn
 * @brief L0C accumulate algo of AvgPooling
 */
typedef enum tagCcPooingFwdAlgo {
  CC_POOLING_FWD_ALGO_HALF = 0,  // accumulate in L0c with FP16
  CC_POOLING_FWD_ALGO_FLOAT32    // accumulate in L0c with FP32
} ccPooingFwdAlgo_t;

/**
 * @ingroup dnn
 * @brief mode of momentum
 */
typedef enum tagMomentumAlgo {
  CC_MOMENTUM_UPDATE_FP32 = 0,  /**< FP32 out */
  CC_MOMENTUM_UPDATE_FP32_FP16, /**< FP32 and FP16 out */
  CC_MOMENTUM_UPDATE_FP32_NESTEROV,
  CC_MOMENTUM_UPDATE_FP32_FP16_NESTEROV,
  CC_MOMENTUM_RESERVED
} ccMomentumAlgo_t;

/**
 * @ingroup dnn
 * @brief mode of partitionStrategy
 *attention: if need to motify this struct,please must motify dPartitionStrategy_t
 */
typedef enum tagCcPartitionStrategy {
  CC_PARTITION_STRATEGY_MOD = 0, /**< mod */
  CC_PARTITION_STRATEGY_DIV,     /**< div */
  CC_PARTITION_STRATEGY_RESERVED
} ccPartitionStrategy_t;

/**
 * @ingroup dnn
 * @brief mode of assignOp
 */
typedef enum tagCcAssignOpMode {
  CC_ASSIGN_ADD = 0, /**< assign add */
  CC_ASSIGN_SUB,     /**< assign sub */
  CC_ASSIGN_RESERVED
} ccAssignOpMode_t;

/**
 * @ingroup dnn
 * @brief mode of arcSinCos
 */
typedef enum tagCcArcSinCosMode {
  CC_ARCUS_SIN = 0, /**< asin */
  CC_ARCUS_COS,     /**< acos */
  CC_ARCUS_RESERVED
} ccArcSinCosMode_t;

/**
 * @ingroup dnn
 * @brief mode of padding
 */
typedef enum tagCcPaddingMode {
  CC_PADDING_CEIL = 0,
  CC_PADDING_DIRECTASSIGN,
  CC_PADDING_VALID,
  CC_PADDING_SAME,      /**< Padding values of 0 are always used */
  CC_PADDING_CEIL_NEW,  /*new ceil,use for backward compatibility*/
  CC_PADDING_VALID_NEW, /*new valid,use for backward compatibility*/
  CC_PADDING_SAME_NEW,  /*new same,use for backward compatibility*/
  CC_PADDING_RESERVED
} ccPaddingMode_t;

/**
 * @ingroup dnn
 * @brief mode of activation
 */
typedef enum tagCcActivationMode {
  CC_ACTIVATION_SIGMOID = 0,  /**< sigmoid */
  CC_ACTIVATION_RELU,         /**< ReLU */
  CC_ACTIVATION_TANH,         /**< tanh */
  CC_ACTIVATION_CLIPPED_RELU, /**< clipped ReLU */
  CC_ACTIVATION_ELU,          /**< ELU */
  CC_ACTIVATION_LEAKY_RELU,
  CC_ACTIVATION_ABS,            /**< Abs */
  CC_ACTIVATION_RELU1,          /**< relu1 */
  CC_ACTIVATION_SOFTSIGN,       /**< softsign */
  CC_ACTIVATION_SOFTPLUS,       /**< softplus */
  CC_ACTIVATION_HARDSIGMOID,    /**< hardsigmoid*/
  CC_ACTIVATION_THRESHOLD_RELU, /**< threshold */
  CC_ACTIVATION_SELU,           /**< selu */
  CC_ACTIVATION_LINEAR,         /**< linear */
  CC_ACTIVATION_RELU6,          /**< relu6 */
  CC_ACTIVATION_RESERVED
} ccActivationMode_t;

/**
 * @ingroup dnn
 * @brief mode of logical op mode
 */
typedef enum tagCcLogicalOpMode {
  CC_LOGICAL_OP_NOT = 0, /**logical not**/
  CC_LOGICAL_OP_AND,     /**logical and**/
  CC_LOGICAL_OP_OR,      /**logical or**/
  CC_LOGICAL_OP_XOR,     /**logical xor**/
  CC_LOGICAL_OP_RESERVED
} ccLogicalOpMode_t;

/**
 * @ingroup dnn
 * @brief mode of batchnorm
 */
typedef enum tagCcBatchNormMode {
  CC_BATCHNORM_PER_ACTIVATION = 0, /**< bnScale, bnBias tensor dims are 1xCxHxW */
  CC_BATCHNORM_SPATIAL,            /**< bnScale, bnBias tensor dims are 1xCx1x1 */
  CC_BATCHNORM_RESERVED
} ccBatchNormMode_t;

/**
 * @ingroup dnn
 * @brief mode of instancenorm
 */
typedef enum tagCcInstanceNormMode {
  CC_INSTANCENORM_PER_ACTIVATION = 0, /**< inScale, inBias tensor dims are NxCxHxW */
  CC_INSTANCENORM_SPATIAL,            /**< inScale, inBias tensor dims are NxCx1x1 */
  CC_INSTANCENORM_RESERVED
} ccInstanceNormMode_t;
/**
 * @ingroup dnn
 * @brief mode of layernorm
 */
typedef enum tagCcLayerNormMode {
  CC_LAYERNORM_PER_ACTIVATION = 0, /**< lnScale, lnBias tensor dims are 1xCxHxW */
  CC_LAYERNORM_SPATIAL,            /**< lnScale, lnBias tensor dims are Nx1x1x1 */
  CC_LAYERNORM_RESERVED
} ccLayerNormMode_t;

/**
 * @ingroup dnn
 * @brief softmax algorithm
 */
typedef enum tagCcSoftmaxAlgo {
  CC_SOFTMAX_FAST = 0,      /**< straightforward implementation */
  CC_SOFTMAX_ACCURATE,      /**< subtract max from every point to avoid overflow */
  CC_SOFTMAX_LOG,           /**< perform the Log softmax operation to avoid overflow */
  CC_SOFTMAX_ACCURATE_FP32, /**< accurate mode for fp32 */
  CC_SOFTMAX_RESERVED
} ccSoftmaxAlgo_t;

/**
 * @ingroup dnn
 * @brief softmax mode
 */
typedef enum tagCcSoftmaxMode {
  CC_SOFTMAX_MODE_INSTANCE = 0, /**< compute the softmax over all C, H, W for each N */
  CC_SOFTMAX_MODE_CHANNEL,      /**< compute the softmax over all C for each H, W, N */
  CC_SOFTMAX_MODE_HEIGHT,       /**< compute the softmax over all H for each N, C, W */
  CC_SOFTMAX_MODE_WIDTH,        /**< compute the softmax over all W for each N, C, H */
  CC_SOFTMAX_MODE_CLASS,        /**< special mode: compute the softmax over all class for each N, H ,W */
  CC_SOFTMAX_MODE_RESERVED
} ccSoftmaxMode_t;

/**
 * @ingroup dnn
 * @brief cross entropy mode
 */
typedef enum tagCcCrossEntropyMode {
  CC_CROSS_ENTROPY_SPARSE_WITHOUT_REDUCTION = 0, /**< compute the sparse cross entropy without fused reduce mean */
  CC_CROSS_ENTROPY_SPARSE_WITH_REDUCTION,        /**< compute the sparse cross entropy with fused reduce mean*/
  CC_CROSS_ENTROPY_WITHOUT_REDUCTION,            /**< compute the cross entropy without fused reduce mean */
  CC_CROSS_ENTROPY_WITH_REDUCTION,               /**< compute the cross entropy with fused reduce mean */
  CC_CROSS_ENTROPY_RESERVED
} ccCrossEntropyMode_t;

/**
 * @ingroup dnn
 * @brief concat mode
 */
typedef enum tagCcConcatMode {
  CC_CONCAT_BY_BATCH = 0, /**< concat by batch */
  CC_CONCAT_BY_FEATURE,   /**< concat by feature */
  CC_CONCAT_BY_HEIGHT,    /**< concat by height */
  CC_CONCAT_BY_WIDTH,     /**< concat by width */
  CC_CONCAT_BY_FLATTEN,
  CC_CONCAT_RESERVED
} ccConcatMode_t;

/**
 * @ingroup dnn
 * @brief eltwise mode
 */
typedef enum tagCcEltwiseMode {
  CC_ELTWISE_PROD = 0, /**< prod */
  CC_ELTWISE_SUM,      /**< sum */
  CC_ELTWISE_MAX,      /**< max */
  CC_ELTWISE_RESERVED
} ccEltwiseMode_t;

/**
 * @ingroup dnn
 * @brief depthwise filter type
 */
typedef enum tagCcDepthwiseFilterType {
  CC_Depthwise_FILTER_DEPTHWISE = 0, /**< depthwise filter */
  CC_Depthwise_FILTER_POINTWISE,     /**< pointwise filter */
  CC_Depthwise_FILTER_RESERVED
} ccDepthwiseFilterType_t;

/**
 * @ingroup dnn
 * @brief sampler type
 */
typedef enum tagCcSamplerType {
  CC_SAMPLER_BILINEAR = 0, /**< bilinear sampler algo */
  CC_SAMPLER_RESERVED
} ccSamplerType_t;

/**
 * @ingroup dnn
 * @brief NMS type
 */
typedef enum tagCcNmsType {
  CC_NMS_IOU = 0, /**< nms operation type, only IOU for now */
  CC_NMS_RESERVED
} ccNmsType_t;

/**
 * @ingroup dnn
 * @brief Box Code type
 */
typedef enum tagCcBoxCodeType {
  CC_BOX_CORNER = 1, /**< Box CodeType in detection nets */
  CC_BOX_CENTER_SIZE,
  CC_BOX_CORNER_SIZE,
  CC_BOX_RESERVED
} ccBoxCodeType_t;

/**
 * @ingroup dnn
 * @brief split mode
 */
typedef enum tagSplitMode {
  CC_SPLIT_MODE_SLICE = 0, /**< spilt data of one dim*/
  CC_SPLIT_MODE_DUPLICATE, /**< copy data of one dim*/
  CC_SPLIT_MODE_RESERVED
} ccSplitMode_t;

/**
 * @ingroup dnn
 * @brief mode of LRN
 */
typedef enum tagCcLRNMode {
  CC_LRN_CROSS_CHANNELS = 0, /**< CROSS_CHANNELS */
  CC_LRN_WITHIN_CHANNELS,    /**< WITHIN_CHANNELS */
  CC_LRN_RESERVED
} ccLRNMode_t;

/**
 * @ingroup dnn
 * @brief format of AIPP input
 */
typedef enum tagCcAippInputFormat {
  CC_AIPP_INPUT_YUV420SP_U8 = 1,
  /**< YUV420SP */  // mini,lite,tiny
  CC_AIPP_INPUT_XRGB8888_U8,
  /**< XRGB8888 */  // mini,lite,tiny
  CC_AIPP_INPUT_NC1HWC0DI_FP16,
  /**< NC1HWC0DI_FP16 */  // mini
  CC_AIPP_INPUT_NC1HWC0DI_S8,
  /**< NC1HWC0DI_S8 */  // mini
  CC_AIPP_INPUT_RGB888_U8,
  /**< RGB888 */  // mini,tiny
  CC_AIPP_INPUT_ARGB8888_U8,
  /**< ARGB8888 */  // lite
  CC_AIPP_INPUT_YUYV_U8,
  /**< YUYV */  // lite
  CC_AIPP_INPUT_YUV422SP_U8,
  /**< YUV422SP */  // lite
  CC_AIPP_INPUT_AYUV444_U8,
  /**< AYUV444 */  // lite
  CC_AIPP_INPUT_YUV400_U8,
  /**< YUV400 */  // mini,lite,tiny
  CC_AIPP_INPUT_RESERVED
} ccAippInputFormat_t;

/**
 * @ingroup dnn
 * @brief mode of AIPP padding
 */
typedef enum tagCcAippPaddingMode {
  CC_AIPP_PAD_DEFAULT_VALUE = 1, /**< CONFIG_VALUE */
  CC_AIPP_PAD_LINE_COPY,         /**< ROW_COL_COPY */
  CC_AIPP_PAD_BLOCK_COPY,        /**< BLOCK_COPY */
  CC_AIPP_PAD_MIRROR_COPY,       /**< MIRROR_COPY */
  CC_AIPP_PAD_RESERVED
} ccAippPaddingMode_t;

/**
 * @ingroup dnn
 * @brief format of cmp type
 */
typedef enum tagCcccCMPType {
  CC_CMP_EQ = 0,
  CC_CMP_NE,
  CC_CMP_LT,
  CC_CMP_GT,
  CC_CMP_GE,
  CC_CMP_LE,
  CC_CMP_TYPE_RESERVED
} ccCMPType_t;

/**
 * @ingroup dnn
 * @brief mode of logical op mode
 */
typedef enum tagCcResultType {
  CC_Result_AND = 0, /**logical and**/
  CC_Result_OR,      /**logical or**/
  CC_Result_RESERVED
} ccResultType_t;

/**
 * @ingroup dnn
 * @brief method of crop_and_resize operator
 */
typedef enum tagCcResizeMethod {
  CC_RESIZE_METHOD_BILINEAR = 0, /** BILINEAR */
  CC_RESIZE_METHOD_NEAREST,      /** NEAREST */
  CC_RESIZE_METHOD_RESERVED
} ccResizeMethod_t;

/**
 * @ingroup dnn
 * @brief mode of calculating new size of the images
 */
typedef enum tagCcResizeOutputDimMode {
  RESIZE_OUTPUT_DIM_BY_ZOOM_FACTOR = 0, /**< Output dimension specified by zoom factor*/
  RESIZE_OUTPUT_DIM_BY_SHRINK_FACTOR,   /**< specified by shrink factor */
  RESIZE_OUTPUT_DIM_EXPLICIT,           /**< specified explicitly */
  RESIZE_OUTPUT_DIM_RESERVED
} ccResizeOutputDimMode_t;

typedef enum tagCcYoloVersion {
  CC_YOLO_V2 = 1, /**< YOLOv2 */
  CC_YOLO_V3,     /**< YOLOv3 */
  CC_YOLO_RESERVED
} ccYoloVersion_t;

typedef enum tagCcAttentionAlgo {
  // bahdanau-attention, for detail:https://pravn.wordpress.com/2017/11/14/bahdanau-attention/
  CC_ATTENTION_ALGO_BAHDANAU = 0,
  CC_ATTENTION_ALGO_NORMAL_BAHDANAU = 1,
  CC_ATTENTION_ALGO_LUONG = 2,
  CC_ATTENTION_ALGO_SCALED_LUONG = 3,
  CC_ATTENTION_ALGO_RESERVED
} AttentionAlgo_t;
/**
 * @ingroup dnn
 * @brief desc of data layout
 */
typedef enum ccEmAttnDecoderDataLayout {
  CC_ATTN_5D_TX1BX,  //[max_time,Xt1,1,batch_size,Xt0]
  CC_ATTN_5D_BTX1X,  //[batch_size*max_time,Xt1,1,1,Xt0]
  CC_ATTN_DL_RESERVED
} ccEmAttnDecoderDataLayout_t;

/**
 * @ingroup dnn
 * @brief operation of Reduce
 */
typedef enum {
  CC_REDUCE_OP_SUM = 0,    /**< sum */
  CC_REDUCE_OP_MEAN,       /**< mean */
  CC_REDUCE_OP_PROD,       /**< product */
  CC_REDUCE_OP_ALL,        /**< logical and */
  CC_REDUCE_OP_ABS_SUM,    /**< absolute sum */
  CC_REDUCE_OP_SQUARE_SUM, /**< square sum */
  CC_REDUCE_OP_MAX,        /**< max */
  CC_REDUCE_OP_MIN,        /**< min */
  CC_REDUCE_OP_LOGSUMEXP,  /**< logsumexp */
  CC_REDUCE_OP_INVALID
} ccReduceOpType_t;

/**
 * @ingroup dnn
 * @brief desc of tpye layout
 */
typedef enum {
  LSH_PROJECTION_TYPE_UNKNOWN = 0,
  LSH_PROJECTION_TYPE_SPARSE = 1,
  LSH_PROJECTION_TYPE_DENSE = 2
} LSHProjectionType;

/**
 * @ingroup dnn
 * @brief activation para
 */
typedef struct tagCcActivationRelu {
  double reluCoef; /* reluCoef for clipped RELU */
  ccNanPropagation_t reluNanOpt;
} ccActivationRelu_t;
typedef union tagCcActivationPara {
  ccActivationRelu_t actionRelu; /* relu Coef and NanOpt for clipped RELU */
  double eluAlpha;               /* eluAlpha for ELU */
  float leakyReluNegativeSlope;
} ccActivationPara_u;

/**
 * @ingroup dnn
 * @bref mode of square
 */
typedef enum tagCcSquareMode {
  CC_SQUARE_2 = 0, /* square */
} ccSquareMode_t;

/**
 * @ingroup dnn
 * @brief append operation type
 */
typedef enum tagCcOpType {
  CC_OP_TYPE_NO_RELU = 0,
  CC_OP_TYPE_RELU = 1,
  CC_OP_TYPE_RELU6 = 2,
  CC_OP_TYPE_INVALID
} ccOpType_t;

/**
 * @ingroup dnn
 * @brief struct define of fill operator type.
 */
typedef enum tagCcFillOpType {
  CC_CONSTANT = 0,
  CC_RANGE,
  CC_LENGTH_RANGE,
  CC_GIVEN_TENSOR,
  CC_DIAGONAL,
  CC_UNIFORM,
  CC_UNIFORM_INT,
  CC_UNIQUE_UNIFORM,
  CC_GAUSSIAN,
  CC_XAVIER,
  CC_MSRA,
  CC_FILL_OP_TYPE_RESERVED
} ccFillOpType_t;

/**
 * @ingroup dnn
 * @brief loss function reduction mode
 */
typedef enum tagCcLossReduction {
  CC_LOSS_REDUCTION_NONE = 0,
  CC_LOSS_REDUCTION_SUM,
  CC_LOSS_REDUCTION_RESERVED
} ccLossReduction_t;

/**
 * @ingroup dnn
 * @brief max size of ccIntArray
 */
#define CC_INT_ARRAY_MAX_SIZE (8)

/**
 * @ingroup dnn
 * @brief struct define of int array less than 8.
 */
typedef struct tagIntArray {
  uint32_t size;
  int32_t value[CC_INT_ARRAY_MAX_SIZE];
} ccIntArray_t;

typedef enum tagCcPadMode {
  CC_PAD_CONSTANT = 0, /*CONSTANT */
  CC_PAD_REFLECT,      /*REFLECT */
  CC_PAD_SYMMETRIC,    /*SYMMETRIC*/
  CC_PAD_EDGE,         /*EDGE */
  CC_PAD_MODE_RESERVED
} ccPadMode_t;

/*
 * @ingroup dnn
 * @brief pad operation of extractImagePatches
 */
typedef enum {
  CC_EXTRACT_IMAGE_PATCHES_PAD_VALID = 1,
  CC_EXTRACT_IMAGE_PATCHES_PAD_SAME,
  CC_EXTRACT_IMAGE_PATCHES_PAD_RESERVED
} ccExtractImagePatchesPadType_t;

/**
 * @ingroup dnn
 * @brief image dimensions of aipp input
 */
#define CC_AIPP_IMG_DIM (2)

/**
 * @ingroup dnn
 * @brief image channel number of aipp input
 */
#define CC_AIPP_IMG_CHN_NUM (4)

/**
 * @ingroup dnn
 * @brief element number of aipp color space convertion matrix
 */
#define CC_AIPP_CSC_MATRIX_DIM (9)

/**
 * @ingroup dnn
 * @brief element number of aipp color space convertion bias
 */
#define CC_AIPP_CSC_BIAS_DIM (3)

/**
 * @ingroup dnn
 * @brief struct define of AIPP operator
 */

typedef struct tagCcAipp {
  ccAippInputFormat_t inputFormat;
  ccDataType_t outputFormat;
  int32_t srcImageSize[CC_AIPP_IMG_DIM];
  int32_t loadStartPos[CC_AIPP_IMG_DIM];
  int32_t loadSize[CC_AIPP_IMG_DIM];
  int32_t scfInputSize[CC_AIPP_IMG_DIM];
  int32_t scfOutputSize[CC_AIPP_IMG_DIM];
  int32_t cscMatrix[CC_AIPP_CSC_MATRIX_DIM];
  int32_t cscOutputBias[CC_AIPP_CSC_BIAS_DIM];
  int32_t cscInputBias[CC_AIPP_CSC_BIAS_DIM];
  int32_t dtcPixelMean[CC_AIPP_IMG_CHN_NUM];
  float dtcPixelMin[CC_AIPP_IMG_CHN_NUM];
  float dtcPixelVarReci[CC_AIPP_IMG_CHN_NUM];
  ccAippPaddingMode_t paddingMode;
  int32_t paddingSize[CC_AIPP_IMG_DIM * 2];  // up,down,left,right
  float cpaddingVaule;
  bool cscSwitch;  // 0:off,1:on
  bool scfSwitch;  // 0:off,1:on
  bool rbuvSwapSwitch;
  bool axSwapSwitch;
  bool singleLineMode;
  bool cscConfigFlag;
  bool dtcConfigFlag;
  bool padConfigFlag;
  bool commConfigFlag;
  bool aippEn;
  bool dyncAippFlag;
  const void *dyncParaAddr;
  bool rotationFlag;
} ccConvolutionAipp_t;


typedef struct tagCcQuantizePara {
  ccConvolutionScaleValueMode_t scaleValueMode;
  uint16_t *scale;
  uint16_t *offsetq;
  int32_t *offsetw;
  uint8_t *allOffsetw;
  uint8_t *offsetPad;
} CcQuantizePara_t;

typedef struct tagCcQuantize {
  ccQuantizeAlgo_t quantAlgo;

  ccConvolutionScaleType_t scaleWType;  // show scaleRq,scaleDq type

  CcQuantizePara_t scaleQ;

  CcQuantizePara_t scaleRq;

  CcQuantizePara_t scaleDq;

  // need relu
  bool reluFlag;

  // relu6
  uint16_t *scaleRelu6;
  bool bConcat;
} ccQuantize_t;

typedef struct tagCcPad *ccPadDescriptor_t;

/**
 * @ingroup dnn
 * @brief operation of Cum
 */
typedef enum {
  CC_CUM_OP_SUM = 0, /**< sum */
  CC_CUM_OP_PROD,    /**< product */
  CC_CUM_OP_INVALID
} CumOpType;

/**
 * @ingroup dnn
 * @brief desciptor of tensor
 */
typedef struct tagCcTensor *ccTensorDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of filter tensor
 */
typedef struct tagCcFilter *ccFilterDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of convolution operator
 */
typedef struct tagCcConvolution *ccConvolutionDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of correlation operator
 */
typedef struct tagCcConvolution *ccCorrelationDescriptor_t;
typedef struct tagCcFullConnection_t *ccFullConnectionDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of pooling operator
 */
typedef struct tagCcPooling *ccPoolingDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of activation operator
 */
typedef struct tagCcActivation *ccActivationDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of batchToSpace operator
 */
typedef struct tagCcBatchToSpace *ccBatchToSpaceDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of spaceToBatch operator
 */
typedef struct tagCcSpaceToBatch *ccSpaceToBatchDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of svdf operator
 */
typedef struct tagCcSvdf *ccSvdfDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of crop operator
 */
typedef struct tagCcCrop *ccCropDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of interp operator
 */
typedef struct tagCcInterp *ccInterpDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of GetRegionBox operator
 */
typedef struct tagCcGetRegionBox *ccGetRegionBoxDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of CorrectBoxes operator
 */
typedef struct tagCorrectBoxes *ccCorrectBoxesDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of ClsProb operator
 */
typedef struct tagClsProb *ccClsProbDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of NMS operator
 */
typedef struct tagCcNms *ccNmsDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of MultiClassNms operator
 */
typedef struct tagCcMultiClassNms *ccMultiClassNmsDescriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of MscnnBoxOutput operator
 */
typedef struct tagCcMscnnBoxOutput *ccMscnnBoxOutputDescriptor_t;

/**
 * @ingroup dnn
 * @brief define of SoftmaxTree
 */
typedef void *ccSoftmaxTree_t;

/**
 * @ingroup dnn
 * @brief descriptor of exp operator
 */
typedef struct tagCcExp *ccExpDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of log operator
 */
typedef struct tagCcLog *ccLogDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of pow operator
 */
typedef struct tagCcPow *ccPowDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of padv2 operator
 */
typedef struct tagCcPadV2 *ccPadV2Descriptor_t;

/**
 * @ingroup dnn
 * @brief desciptor of ShapeClassify operator
 */
typedef struct tagCcShapeClassify *ccShapeClassifyDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of DetectionFull3DOutput operator
 */
typedef struct tagCcDetectionFull3DOutput *ccDetectionFull3DOutputDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of Quantize operator
 */
typedef struct tagCcQuantize *ccQuantizeDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of StridedSlice operator
 */
typedef struct tagCcStridedSlice *ccStridedSliceDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of StridedSliceAttrs operator
 */
typedef struct tagCcStridedSliceAttrs *ccStridedSliceAttrsDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of ResizeBilinear operator
 */
typedef struct tagCcResizeBilinear *ccResizeBilinearDescriptor_t;

typedef struct tagCcEltwise *ccEltwiseDescriptor_t;

typedef struct tagCcBatchNorm *ccBatchNormDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of Square operator
 */
typedef struct tagCcSquare *ccSquareDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of NonMaxSuppression operator
 */
typedef struct tagNonMaxSuppression *ccNonMaxSuppressionDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of NonMaxSuppression operator
 */
typedef struct tagUpsamplePara *ccUpsampleParaDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of ResizeNearestNeighbor operator
 */
typedef struct tagCcResizeNearestNeighbor *ccResizeNearestNeighborDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of Fill operator
 */
typedef struct tagCcFillParam *ccFillParamDescriptor_t;

/**
 * @ingroup dnn
 * @brief descriptor of Argmaxmin operator
 */
typedef struct tagCcArgmaxmin *ccArgmaxminDescriptor_t;

};  // namespace cce

#endif  // DNN_BASE_HPP__

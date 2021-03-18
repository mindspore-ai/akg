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

#ifndef DNN_STRUCT_BASE_HPP__
#define DNN_STRUCT_BASE_HPP__

#include "cce/cce_def.hpp"

namespace cce {

/**
 * @ingroup dnn
 * @brief max number of dimensions
 */
#define CC_DIM_MAX (8)

/**
 * @ingroup dnn
 * @brief max number of dimensions when use NC1HWC0 format
 */
#define CC_REALDIM_MAX (4)

/**
 * @ingroup dnn
 * @brief max input count of MscnnBoxOutput
 */
#define CC_MAX_INPUT_CNT (10)

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
 * @brief parameter number of op exp/log/pow
 */
#define PARAM_CNT_THREE (3)

/**
 * @ingroup dnn
 * @brief parameter number of op nonmaxsuppression
 */
#define PARAM_CNT_TWO (2)
#define DIMCNT_NUMBER_ONE (1)
#define DIMCNT_NUMBER_TWO (2)
#define DIMCNT_NUMBER_FOUR (4)

#define COMMON_FORMAT_NCHW_N_INDEX (0)
#define COMMON_FORMAT_NCHW_C_INDEX (1)
#define COMMON_FORMAT_NCHW_H_INDEX (2)
#define COMMON_FORMAT_NCHW_W_INDEX (3)

/**
 * @ingroup dnn
 * @brief parameter number of op upsample
 */
#define UPSAMPLE_SCAL_DEFAULT_TWO (2)
#define UPSAMPLE_ILLEGAL_VALUE_1 (1)

/**
 * @ingroup dnn
 * @brief struct define of StridedSlice required params.
 */

typedef struct tagCcStridedSlice {
  uint32_t dimCnt;
  int32_t begin[CC_DIM_MAX];
  int32_t end[CC_DIM_MAX];
  int32_t strides[CC_DIM_MAX];
} ccStridedSlice_t;

/**
 * @ingroup dnn
 * @brief struct define of Strided_slice attrs
 */
typedef struct tagCcStridedSliceAttrs {
  uint32_t beginMask;
  uint32_t endMask;
  uint32_t ellipsisMask;
  uint32_t newAxisMask;
  uint32_t shrinkAxisMask;
} ccStridedSliceAttrs_t;

/**
 * @ingroup dnn
 * @brief params of batchToSpace
 */
typedef struct tagCcBatchToSpace {
  int32_t blockShapeLength;
  int32_t blockShape[CC_DIM_MAX];
  int32_t crops[2 * CC_DIM_MAX];
} ccBatchToSpace_t;

/**
 * @ingroup dnn
 * @brief params of spaceToBatch
 */
typedef struct tagCcSpaceToBatch {
  int32_t blockShapeLength;
  int32_t blockShape[CC_DIM_MAX];
  int32_t paddings[2 * CC_DIM_MAX];
} ccSpaceToBatch_t;

/**
 * @ingroup dnn
 * @brief struct define of tensor
 */
typedef struct tagCcTensor {
  ccTensorFormat_t format;
  ccDataType_t dataType;
  int32_t dimCnt;
  int32_t realDimCnt;
  uint32_t dataSize;
  int32_t dim[CC_DIM_MAX];
  int32_t stride[CC_DIM_MAX];
  ccVecQuantizePara_t vecQuantizePara;
} ccTensor_t;

/**
 * @ingroup dnn
 * @brief struct define of filter tensor
 */
typedef struct tagCcFilter {
  ccTensorFormat_t format;
  ccDataType_t dataType;
  int32_t dimCnt;
  uint32_t dataSize;
  int32_t dim[CC_DIM_MAX];
} ccFilter_t;

/**
 * @ingroup dnn
 * @brief struct define of convolution operator
 */
typedef struct tagCcConvolution {
  ccConvolutionMode_t mode;
  ccPaddingMode_t padMode;
  int32_t dimCnt;
  int32_t padding[2 * (CC_DIM_MAX - 2)];
  int32_t filterStride[CC_DIM_MAX - 2];
  int32_t dilation[CC_DIM_MAX - 2];
  int32_t group;
  ccQuantizeDescriptor_t quantInfo;
  ccConvolutionAipp_t aippInfo;
  int32_t adj[CC_DIM_MAX - 2];
  int32_t targetShape[CC_DIM_MAX - 2];
  int32_t beforePadding[2 * (CC_DIM_MAX - 2)];  // pad before conv
  uint32_t reluFlag;
  int64_t concatBatchSize;
} ccConvolution_t;

#define ccCorrelation_t ccConvolution_t
typedef struct tagCcFullConnection_t {
  ccQuantizeDescriptor_t quantInfo;
  uint32_t infoTabSize;
  const void *infoTab;
  bool reluFlag;
  ccFullConnectFwdAlgo_t algo;
} ccFullConnection_t;

typedef struct tagCcConcatFour2Five_t {
  uint32_t branchNum;  // how many branch for box or class
  uint32_t classNum;   // box branch's classNum is four, class branch's classNum is class number
} ccConcatFour2Five_t;

typedef struct tagCcTransdata_t {
  uint64_t scaleQAddr;
  uint8_t scaleQValueMode;
  uint64_t offsetQAddr;
  uint8_t quantAlgo;
  uint8_t quantize8bitFlag;
} ccTransdata_t;
/**
 * @ingroup dnn
 * @brief struct define of pooling operator
 */
typedef struct tagCcPooling {
  ccPoolingMode_t mode;
  ccPaddingMode_t padMode;
  ccNanPropagation_t maxpoolingNanOpt;
  int32_t dimCnt;
  int32_t windowDim[CC_DIM_MAX - 2];
  int32_t padding[CC_DIM_MAX - 2];
  int32_t stride[CC_DIM_MAX - 2];
  int32_t dataMode;
  int32_t ceilMode;
  ccQuantizeDescriptor_t quantInfo;
  ccPooingFwdAlgo_t algo;
} ccPooling_t;

/**
 * @ingroup dnn
 * @brief struct define of activation operator
 */
typedef struct tagCcActivation {
  ccActivationMode_t mode;
  ccNanPropagation_t reluNanOpt;
  double coef; /* ceiling for clipped RELU, alpha for ELU */
  ccActivationPara_u activationPara;
} ccActivation_t;

/**
 * @ingroup dnn
 * @brief struct define of svdf operator
 */
typedef struct tagCcSvdf {
  ccTensorFormat_t format;
  ccDataType_t dataType;
  uint32_t batches;
  uint32_t features;
  uint32_t rank;
  uint32_t inputSize;
  uint32_t memorySize;
} ccSvdf_t;

/**
 * @ingroup dnn
 * @brief struct define of svdf operator
 */
typedef struct tagCcHashTableLookup {
  ccTensorFormat_t format;
  ccDataType_t lookupType;
  ccDataType_t keyType;
  ccDataType_t valueType;
  ccDataType_t outputType;
  ccDataType_t hitsType;
  uint32_t lookups;
  uint32_t keys;
  uint32_t rows;
  uint32_t features;
  uint16_t valueScale;
  uint16_t outputScale;
  uint16_t valueOffset;
  uint16_t outputOffset;
} ccHashTableLookup_t;

/**
 * @ingroup dnn
 * @brief struct define of prelu operator
 */
typedef struct tagCcPRelu {
  ccNanPropagation_t reluNanOpt;
  int32_t slopeCount;
  bool channelShared;
} ccPRelu_t;

/**
 * @ingroup dnn
 * @brief struct define of crop operator
 */
typedef struct tagCcCrop {
  int32_t startAxis;
  int32_t offset[CC_DIM_MAX];
  int32_t offsetCnt;
} ccCrop_t;

/**
 * @ingroup dnn
 * @brief struct define of SpatialTransformer operator
 */
typedef struct tagCcSpatialTransformer {
  ccSamplerType_t samplerType;
  ccDataType_t dataType;
  int32_t dimCnt;
  uint64_t dim[CC_DIM_MAX];
  uint64_t alignCorner;
} ccSpatialTransformer_t;

/**
 * @ingroup dnn
 * @brief struct define of ShiftTransformer operator
 */
typedef struct tagCcShiftTransformer {
  ccSamplerType_t samplerType;
  double xPreDefined;
  double yPreDefined;
  bool xShift;
  bool yShift;
  int32_t gridH;
  int32_t gridW;
} ccShiftTransformer_t;

/**
 * @ingroup dnn
 * @brief struct define of FasterRcnnProposal operator
 */
typedef struct tagCcFasterRcnnProposal {
  int32_t preNMStopK;
  int32_t postNMStopK;
  float nmsTresh;
  float minSize;
  float featStride;
  float baseSize;
  int32_t ratioCnt;
  int32_t scaleCnt;
  float *ratio;
  float *scale;
  int32_t imgH;
  int32_t imgW;
} ccFasterRcnnProposal_t;

/**
 * @ingroup dnn
 * @brief struct define of LRN operator
 */
typedef struct tagCcLRN {
  ccLRNMode_t lrnMode;
  int32_t lrnN;
  double lrnAlpha;
  double lrnBeta;
  double lrnK;
} ccLRN_t;

/**
 * @ingroup dnn
 * @brief struct define of instanceNorm
 */
typedef struct tagCcInstancenorm {
  ccInstanceNormMode_t mode;
  double epsilon;
} ccInstancenorm_t;

/**
 * @ingroup dnn
 * @brief struct define of assignOp operator
 */
typedef struct tagCcAssignOp {
  ccAssignOpMode_t assignOpMode;
} ccAssignOp_t;

/**
 * @ingroup dnn
 * @brief struct define of arcSinCos operator
 */
typedef struct tagCcArcSinCos {
  ccArcSinCosMode_t arcSinCosMode;
} ccArcSinCos_t;

/**
 * @ingroup dnn
 * @brief struct define of Detectpostprocess operator
 */
typedef struct tagCcDetectpostprocess {
  int32_t numClasses;
  float confThreshold;
  float nmsThreshold;
  int32_t outTopK;
  float bboxRegWeightsDx;
  float bboxRegWeightsDy;
  float bboxRegWeightsDw;
  float bboxRegWeightsDh;
} ccDetectpostprocess_t;
/**
 * @ingroup dnn
 * @brief struct define of FasterRcnnDetectionOutput operator
 */
typedef struct tagCcFasterRcnnDetectionOutput {
  int32_t numClasses;
  float nmsThreshold;
  float postConfThreshold;
  int32_t imgH;
  int32_t imgW;
  int32_t batchSize;
} ccFasterRcnnDetectionOutput_t;

/**
 * @ingroup dnn
 * @brief struct define of SsdDetectionOutput operator
 */
typedef struct tagCcSsdDetectionOutput {
  int32_t numClasses;
  int32_t backgroundLabelId;
  double preConfThreshold;
  int32_t preTopK;
  double nmsThreshold;
  double nmsEta;
  ccBoxCodeType_t codeType;
  int32_t outTopK;
  bool shareLocation;
  bool varianceEncodedInTarget;
  uint32_t boxTypeNum;
  float var[4];
  uint32_t variance_num;
} ccSsdDetectionOutput_t;

/**
 * @ingroup dnn
 * @brief struct define of RefinedetDetectionOutput operator
 */
typedef struct tagCcRefinedetDetectionOutput {
  int32_t numClasses;
  int32_t backgroundLabelId;
  double preConfThreshold;
  int32_t preTopK;
  double nmsThreshold;
  double nmsEta;
  ccBoxCodeType_t codeType;
  int32_t outTopK;
  bool shareLocation;
  bool varianceEncodedInTarget;
  uint32_t boxTypeNum;
  float var[4];
  uint32_t variance_num;
  double objectness_score;
} ccRefinedetDetectionOutput_t;

/**
 * @ingroup dnn
 * @brief struct define of MsrGenerateRpnProposals operator
 */
typedef struct tagCcMsrGenerateRpnProposals {
  int32_t preNmsTopK;
  int32_t postNmsTopK;
  float nmsThreshold;
  float rpnMiniSize;
  int32_t imgH;
  int32_t imgW;
  uint32_t boxTypeNum;
  float scoreThreshold;
} ccMsrGenerateRpnProposals_t;

/**
 * @ingroup dnn
 * @brief struct define of RetinaPostprocessor operator
 */
typedef struct tagCcRetinaPostprocessor {
  int32_t numClasses;
  int32_t maxDetections;
  float nmsThreshold;
  float scoreThreshold;
  int32_t imgH;
  int32_t imgW;
  uint32_t boxTypeNum;
  float mean[4];
  int32_t meanNum;
  float std[4];
  int32_t stdNum;
  int32_t outputNum;
  bool ocrFlag;
} ccRetinaPostprocessor_t;

/**
 * @ingroup dnn
 * @brief struct define of GenerateSsdAnchors operator
 */
typedef struct tagCcGenerateSsdAnchors {
  int32_t featureMapShapeList[20];
  uint32_t featureMapShapeListSize;
  int32_t boxSpecsNum[10];
  uint32_t boxSpecsNumSize;
  float scales[10];
  uint32_t scalesNum;
  float aspectRatios[10];
  uint32_t aspectRatiosNum;
  int32_t baseAnchorSize[2];
  uint32_t baseAnchorSizeNum;
  int32_t anchorStride[2];
  uint32_t anchorStrideNum;
  int32_t anchorOffset[2];
  uint32_t anchorOffsetNum;
  bool reduceBoxesInLowestLayer;
  float minScale;
  float maxScale;
  int32_t imgH;
  int32_t imgW;
} ccGenerateSsdAnchors_t;

/**
 * @ingroup dnn
 * @brief struct define of MscnnBoxOutput operator
 */
typedef struct tagCcMscnnBoxOutput {
  double fgThreshold;
  double nmsThreshold;
  ccNmsType_t nmsType;
  int32_t fieldH[CC_MAX_INPUT_CNT];
  int32_t fieldW[CC_MAX_INPUT_CNT];
  int32_t downsampleRate[CC_MAX_INPUT_CNT];
  int32_t defaultBoxCnt;
  double fieldWhr;
  double fieldXyr;
  int32_t maxNmsNum;
  int32_t maxPostNmsNum;
  double minSize;
} ccMscnnBoxOutput_t;

/**
 * @ingroup dnn
 * @brief struct define of NMS operator
 */
typedef struct tagCcNms {
  int32_t numClasses;
  int32_t backgroundLabelId;
  double preConfThreshold;
  int32_t preTopK;
  double nmsThreshold;
  double nmsEta;
  int32_t postTopK;
  int32_t outTopK;
  double postConfThreshold;
  bool shareLocation;
} ccNms_t;

/**
 * @ingroup dnn
 * @brief struct define of NMS/MultiClassNMS operator
 */
typedef struct tagCcMultiClassNms {
  uint64_t numClasses;
  float objThreshold;
  float nmsThreshold;
  float clsThreshold;
  bool normal;
  uint64_t coorType;
} ccCcMultiClassNms_t;

/**
 * @ingroup dnn
 * @brief struct define of YoloDetectionOutput operator
 */
typedef struct tagCcYoloDetectionOutput {
  ccYoloVersion_t yoloVersion;
  uint32_t netH;
  uint32_t netW;
  uint32_t postTopK;
  uint32_t classes;
  float nmsThreshold;
  float iouThreDecay;
  float coorScaleFactor;
  bool relative;
  float objThreshold;
  float clsThreshold;
  uint32_t biasNum;
  float *bias;
} ccYoloDetectionOutput_t;

/**
 * @ingroup dnn
 * @brief struct define of GetRegionBox operator
 */
#ifndef CC_MAX_YOLO_BIAS_NUM
#define CC_MAX_YOLO_BIAS_NUM (16)
#endif

typedef struct tagCcGetRegionBox {
  uint32_t biasNum;
  uint32_t H;
  uint32_t W;
  float bias[CC_MAX_YOLO_BIAS_NUM];
} ccGetRegionBox_t;

/**
 * @ingroup dnn
 * @brief struct define of CorrectBoxes operator
 */
typedef struct tagCorrectBoxes {
  uint32_t netW;
  uint32_t netH;
  bool relative;
} ccCorrectBoxes_t;

/**
 * @ingroup dnn
 * @brief struct define of ClsProb operator
 */
typedef struct tagClsProb {
  float objThreshold;
} ccClsProb_t;

/**
 * @ingroup dnn
 * @brief struct define of SsdPriorBox operator
 */
typedef struct tagCcSsdPriorBox {
  ccBoxCodeType_t codeType;
  double *minSize;
  int32_t minSizeNum;
  double *maxSize;
  int32_t maxSizeNum;
  double *aspectRatio;
  int32_t aspectRatioNum;
  double *variance;
  int32_t varianceNum;
  int32_t imgH;
  int32_t imgW;
  double stepH;
  double stepW;
  double offset;
  bool flip;
  bool clip;
} ccSsdPriorBox_t;

/**
 * @ingroup dnn
 * @brief struct define of Yolo2Region operator
 */
typedef struct tagCcYolo2Region {
  ccSoftmaxTree_t softmaxTree;
  bool softmax;
  bool background;
  bool treeSoftmax;
} ccYolo2Region_t;

/**
 * @ingroup dnn
 * @brief struct define of YoloRegion operator
 */
typedef struct tagCcYoloRegion {
  ccSoftmaxTree_t softmaxTree;
  bool softmax;
  bool background;
  bool treeSoftmax;
  int32_t classes;
  int32_t coords;
  int32_t boxes;
  ccYoloVersion_t yoloV;
} ccYoloRegion_t;

/**
 * @ingroup dnn
 * @brief struct define of power operator
 */
typedef struct tagCcPower {
  float scale;
  float shift;
  float power;
} ccPower_t;

/**
 * @ingroup dnn
 * @brief struct define of exp operator
 */
typedef struct tagCcExp {
  ccDataType_t dataType;
  uint32_t paramCnt;
} ccExp_t;

/**
 * @ingroup dnn
 * @brief struct define of exp operator
 */
typedef struct tagCcLog {
  ccDataType_t dataType;
  uint32_t paramCnt;
} ccLog_t;

/**
 * @ingroup dnn
 * @brief struct define of pow operator
 */
typedef struct tagCcPow {
  ccDataType_t dataType;
  uint32_t paramCnt;
} ccPow_t;

/**
 * @ingroup dnn
 * @brief struct define of padv2 operator
 */
typedef struct tagCcPadV2 {
  ccPadMode_t padMode;
  void *padValue;
  ccDataType_t padValueType;
  int32_t padDimCnt;
  int32_t padShapeLow[CC_DIM_MAX];
  int32_t padShapeHigh[CC_DIM_MAX];
} ccPadV2_t;

/**
 * @ingroup dnn
 * @brief struct define of psROIPooling operator
 */
typedef struct tagCcPsRoiPooling {
  ccPoolingMode_t poolingMode;
  int32_t pooledH;
  int32_t pooledW;
  float spatialScale;
  float padRatio;
  int32_t groupSize;
  int32_t outputDim;
} ccPsRoiPooling_t;

/**
 * @ingroup dnn
 * @brief struct define of RoIAlign operator
 */
typedef struct tagCcRoiAlign {
  int32_t pooledH;
  int32_t pooledW;
  float spatialScale;
  int32_t samplingRatio;
} ccRoiAlign_t;

/**
 * @ingroup dnn
 * @brief struct define of RoiInterpPooling operator
 */
typedef struct tagCcRoiInterpPooling {
  int32_t pooledH;
  int32_t pooledW;
  int32_t poolKernelH;
  int32_t poolKernelW;
  int32_t pooledTailH;
  int32_t pooledTailW;
  float spatialScaleH;
  float spatialScaleW;
} ccRoiInterpPooling_t;

/**
 * @ingroup dnn
 * @brief struct define of DetectionFull3DOutput operator
 */
typedef struct tagCcDetectionFull3DOutput {
  int32_t imageWidth;
  int32_t imageHeight;
  int32_t numAngleBins;
  float trcMarginRatioX;
  float trcMarginRatioY;
  int32_t pitchRangeD;
  int32_t pitchPresetD;
  float mountHeight;
  int32_t visiblenessBins;
  float meanVisibleness;
  bool discreteVisibleness;
} ccDetectionFull3DOutput_t;

/**
 * @ingroup dnn
 * @brief struct define of MsrFastRcnnPredictions operator
 */
typedef struct tagMsrFastRcnnPredictions {
  int32_t numClasses;    // num of classes
  float scoreThreshold;  // the threshold of the score
  double nmsThreshold;   // the threshold of nms
  int32_t postTopK;
  int32_t outTopK;
  int32_t imgH;  // the height of image
  int32_t imgW;  // the width of image
} ccMsrFastRcnnPredictions_t;

typedef struct tagCcResizeBilinear {
  ccResizeOutputDimMode_t resizeOutputDimMode;
  bool alignCorners;
  int32_t zoom_factor;
  int32_t shrink_factor;
  int32_t height;
  int32_t width;
  int32_t pad_begin;
  int32_t pad_end;
} ccResizeBilinear_t;

typedef struct tagCcResizeNearestNeighbor {
  bool alignCorners;
  int32_t height;
  int32_t width;
} ccResizeNearestNeighbor_t;

typedef struct tagCcEltwise {
  ccQuantize_t *quantInfo;
  bool reluFlag;
} ccEltwise_t;

typedef struct tagCcBatchNorm {
  bool reluFlag;
} ccBatchNorm_t;

typedef struct tagCcPad {
  ccPadMode_t padMode;
  float padValue;
  int32_t htoppad;     // padLow[0]
  int32_t hbottompad;  // padHigh[0]
  int32_t wleftpad;    // padLow[1]
  int32_t wrightpad;   // padHigh[1]
} ccPad_t;

typedef struct tagCcSubCondition {
  uint32_t BaseCondValue[4];
  ccCMPType_t condType[4];
  ccResultType_t resultType;
} ccSubCondition;

typedef struct tagCcShapeClassifyCond {
  uint32_t subConditionNum;
  ccResultType_t resultType;
  uint32_t true_value;
  ccSubCondition subCond[2];
} ccShapeClassifyCond;

#ifndef CC_SHAPE_CLASSIFY_CONDITION_NUM
#define CC_SHAPE_CLASSIFY_CONDITION_NUM (8)
#endif

typedef struct tagCcShapeClassify {
  uint32_t shapeClassifyConditionNum;
  uint32_t defaultValue;
  ccShapeClassifyCond shapeClassifyCond[CC_SHAPE_CLASSIFY_CONDITION_NUM];
} ccShapeClassify_t;

/**
 * @ingroup dnn
 * @bref struct define of square operator
 */
typedef struct tagCcSquare {
  ccSquareMode_t mode;
} ccSquare_t;

/*
 * @ingroup dnn
 * @brief operation of segment reduction
 */
typedef enum {
  CC_SEGMENT_REDUCTION_OP_SUM = 0, /**< sum */
  CC_SEGMENT_REDUCTION_OP_INVALID
} ccSegmentReductionOpType_t;

typedef struct tagCcFillParam {
  // The filler type.
  ccFillOpType_t fillType;
  ccDataType_t valueDatatype;
  const void *value;  // the value in constant fill
  const void *min;    // the min value in uniform fill
  const void *max;    // the max value in uniform fill
  const void *mean;   // the mean value in Gaussian fill
  const void *std;    // the std value in Gaussian fill
  // the seed used to generate data in Gaussian and uniform fill
  int64_t seed1;
  int64_t seed2;
} ccFillParam_t;

typedef struct tagNonMaxSuppression {
  ccDataType_t dataType;
  uint32_t paraCount;
} ccNonMaxSuppression_t;

typedef struct tagCcArgmaxmin {
  int32_t axisType;
  bool outMaxVal;
  int64_t topK;
  int64_t reduceSize;
  int64_t reduceStride;
  int64_t axis;
  bool keepDims;
} ccArgmaxmin_t;

typedef struct tagUpsamplePara {
  int32_t scale;
  int32_t scaleHeight;
  int32_t scaleWidth;
  int32_t upsampleHeight;
  int32_t upsampleWidth;
  bool padOutHeight;
  bool padOutWidth;
} ccUpsamplePara_t;

typedef struct tagCcConcatFive2Four_t {
  ccTransForLossMode_t mode;
  uint32_t classNum;
} ccConcatFive2Four_t;

};     // namespace cce
#endif  // DNN_STRUCT_BASE_HPP__

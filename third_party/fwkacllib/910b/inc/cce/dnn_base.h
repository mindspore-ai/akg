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

#ifndef DNN_BASE_H__
#define DNN_BASE_H__

#include "cce/blas_struct.h"
#include "cce/customize.h"
#include "cce/dnn_base_def.hpp"

namespace cce {
/**
  * @ingroup dnn
  * @brief Minimum epsilon allowed to be used in the Batch Normalization formula
  */
#define CC_BN_MIN_EPSILON               (1e-7)

#ifndef NULL
    #ifdef __cplusplus
        #define NULL 0
    #else
        #define NULL ((void *)0)
    #endif
#endif

/**
  * @ingroup dnn
  * @brief max number of dimensions
  */
#define CC_DIM_MAX (8)

typedef  struct cCTagL2LossDescriptor *   ccL2LossDescriptor_t;

/**
  * @ingroup dnn
  * @brief mode of concatfive2fout
  */
typedef enum tagTransForLossMode {
    CC_TRANS_FOR_BOX = 0,
    CC_TRANS_FOR_SCORE,
} ccTransForLossMode_t;

/**
  * @ingroup dnn
  * @brief descriptor of concatfive2fout
  */
typedef struct tagCcConcatFive2Four_t *ccConcatFive2FourDescriptor_t;

}; /* end cce */

namespace cce {

/**
  * @ingroup dnn
  * @brief create descriptor of tensor
  * @param [in|out] tensorDesc   point to descriptor of tensor
  * @return ccStatus_t
  */
ccStatus_t ccCreateTensorDescriptor(ccTensorDescriptor_t *tensorDesc);

/**
  * @ingroup dnn
  * @brief destroy descriptor of tensor
  * @param [in] *tensorDesc   descriptor of tensor
  * @return ccStatus_t
  */
ccStatus_t ccDestroyTensorDescriptor(ccTensorDescriptor_t *tensorDesc);

/**
  * @ingroup dnn
  * @brief init tensor to 4d tensor
  * @param [in|out] tensorDesc   descriptor of tensor
  * @param [in] format   format of tensor
  * @param [in] dataType   data type in device
  * @param [in] n   batch size
  * @param [in] c   channels
  * @param [in] h   height of feature map
  * @param [in] w   width of feature map
  * @return ccStatus_t
  */
ccStatus_t ccSetTensor4dDescriptor(ccTensorDescriptor_t tensorDesc,
                                   ccTensorFormat_t format,
                                   ccDataType_t dataType,
                                   int32_t n,
                                   int32_t c,
                                   int32_t h,
                                   int32_t w);

/**
  * @ingroup dnn
  * @brief read 4d tensor
  * @param [in] tensorDesc   descriptor of tensor
  * @param [in|out] dataType   point to data type in device
  * @param [in|out] n   point to batch size
  * @param [in|out] c   point to channels
  * @param [in|out] h   point to height of feature map
  * @param [in|out] w   point to width of feature map
  * @param [in|out] nStride   point to stride of n
  * @param [in|out] cStride   point to stride of c
  * @param [in|out] hStride   point to stride of h
  * @param [in|out] wStride   point to stride of w
  * @return ccStatus_t
  */
ccStatus_t ccGetTensor4dDescriptor(const ccTensorDescriptor_t tensorDesc,
                                   ccDataType_t *dataType,
                                   int32_t *n,
                                   int32_t *c,
                                   int32_t *h,
                                   int32_t *w,
                                   int32_t *nStride,
                                   int32_t *cStride,
                                   int32_t *hStride,
                                   int32_t *wStride);

/**
* @ingroup dnn
* @brief print 4d tensor (just in debug log mode)
* @param [in] tensorDesc   descriptor of tensor
* @return ccStatus_t
*/
ccStatus_t ccPrintTensor4dDescriptor(const ccTensorDescriptor_t tensorDesc);

/**
* @ingroup dnn
* @brief print Nd tensor (just in debug log mode)
* @param [in] tensorDesc   descriptor of tensor
* @return ccStatus_t
*/
ccStatus_t ccPrintTensorNdDescriptor(const ccTensorDescriptor_t tensorDesc);

/**
  * @ingroup dnn
  * @brief init tensor to Nd tensor
  * @param [in|out] tensorDesc   descriptor of tensor
  * @param [in] dataType   data type in device
  * @param [in] dimCnt   Dimension of the tensor
  * @param [in] dimA   Array of dimension dimCnt that contain the size of the tensor for every dimension. Size along unused dimensions should be set to 1.
  * @return ccStatus_t
  */
ccStatus_t ccSetTensorNdDescriptor(ccTensorDescriptor_t tensorDesc,
                                   ccDataType_t dataType,
                                   int32_t dimCnt,
                                   int32_t dimA[]);

/**
  * @ingroup dnn
  * @brief read Nd tensor
  * @param [in] tensorDesc   descriptor of tensor
  * @param [in] dimCntReq   point to data type in device
  * @param [in|out] dataType   point to data type in device
  * @param [in|out] dimCnt   Dimension of the tensor
  * @param [in|out] dimA   Array of dimension of at least dimCntReq that will be filled with the dimensions from the provided tensor descriptor.
  * @param [in|out] strideA   Array of dimension dimCntReq that contain the stride of the tensor for every dimension
  * @return ccStatus_t
  */
ccStatus_t ccGetTensorNdDescriptor(const ccTensorDescriptor_t tensorDesc,
                                   int32_t dimCntReq,
                                   ccDataType_t *dataType,
                                   int32_t *dimCnt,
                                   int32_t dimA[],
                                   int32_t strideA[]);

/**
  * @ingroup dnn
  * @brief transform tensor between 4d(NCHW) and 5d(NC1HWC0)
  * @param [in] xDesc   descriptor of input tensor
  * @param [in] x   point to input data in host memory
  * @param [in] dataTypeTransmode   mode of data type transform
  * @param [in] yDesc   descriptor of output tensor
  * @param [in|out] y   point to output data in host memory
  * @param [in] ySizeInBytes   size of outputData
  * @return ccStatus_t
  */
ccStatus_t ccTransTensor(const ccTensorDescriptor_t xDesc,
                         const void *x,
                         const ccTensorDescriptor_t yDesc,
                         void *y,
                         uint32_t ySizeInBytes);

/**
  * @ingroup dnn
  * @brief get the format and dimcnt of Tensor
  * @param [in] tensorDesc   descriptor of tensor
  * @param [in|out] format   point to format
  * @return ccStatus_t
  */
ccStatus_t ccGetTensorFormat(const ccTensorDescriptor_t tensorDesc,
                             ccTensorFormat_t  *format);

/**
  * @ingroup dnn
  * @brief set the format and dimcnt of Tensor
  * @param [in] tensorDesc   descriptor of tensor
  * @param [in|out] format   point to format
  * @return ccStatus_t
  */
ccStatus_t ccSetTensorFormat(ccTensorDescriptor_t tensorDesc,
                             ccTensorFormat_t  format);


/**
  * @ingroup dnn
  * @brief get the RealDimCnt of Tensor
  * @param [in] tensorDesc   descriptor of tensor
  * @param [in|out] RealDimCnt   point to RealDimCnt
  * @return ccStatus_t
  */
ccStatus_t ccGetTensorRealDimCnt(const ccTensorDescriptor_t tensorDesc,
                                 int32_t *realDimCnt);

/**
  * @ingroup dnn
  * @brief set the RealDimCnt of Tensor
  * @param [in|out] tensorDesc   descriptor of tensor
  * @param [in] RealDimCnt   RealDimCnt to set
  * @return ccStatus_t
  */
ccStatus_t ccSetTensorRealDimCnt(ccTensorDescriptor_t tensorDesc,
                                 int32_t realDimCnt);


/**
  * @ingroup dnn
  * @brief get data size of 4d tensor
  * @param [in] tensorDesc   descriptor of tensor
  * @param [in|out] size   point to data size
  * @return ccStatus_t
  */
ccStatus_t ccGetTensorSizeInBytes(const ccTensorDescriptor_t tensorDesc, uint32_t *size);

/**
* @ingroup dnn
* @brief get data size of 4d tensor which is align to 32B
* @param [in] tensorDesc   descriptor of tensor
* @param [in|out] size   point to data size
* @return ccStatus_t
*/
ccStatus_t ccGetTensorMemorySizeInBytes(const ccTensorDescriptor_t tensorDesc, uint32_t *size);


ccStatus_t ccSetTensorDataSize(ccTensorDescriptor_t xDesc, uint32_t size);

/**
  * @ingroup dnn
  * @brief get data size of 4d filter
  * @param [in] filterDesc   descriptor of filter
  * @param [in] groupNum number of group
  * @param [in|out] size   point to data size
  * @return ccStatus_t
  */
ccStatus_t ccGetFilterSizeInBytes(const ccFilterDescriptor_t filterDesc, uint32_t *size);


/**
  * @ingroup dnn
  * @brief read 4d filter
  * @param [in] filterDesc   descriptor of filter
  * @param [in|out] format   point to format of filter
  * @param [in|out] dataType   point to data type in device
  * @param [in|out] k   point to number of output feature maps
  * @param [in|out] c   point to number of input feature maps
  * @param [in|out] h   point to height of filter
  * @param [in|out] w   point to width of filter
  * @return ccStatus_t
  */
ccStatus_t ccGetFilter4dDescriptor(const ccFilterDescriptor_t filterDesc,
                                   ccTensorFormat_t *format,
                                   ccDataType_t *dataType,
                                   int32_t *k,
                                   int32_t *c,
                                   int32_t *h,
                                   int32_t *w);

ccStatus_t ccTransFilterFracZToNCHW(const ccFilterDescriptor_t wDesc,
                                    const void *w,
                                    ccFilterDescriptor_t yDesc,
                                    void *y,
                                    uint32_t ySizeInBytes);

/**
  * @ingroup dnn
  * @brief trans weight to fractal format, and trans data type together
  * @param [in] wDesc   descriptor of input filter
  * @param [in] w   input data pointer
  * @param [in] yDesc   descriptor of output filter
  * @param [in|out] y   output data pointer
  * @param [in] ySizeInBytes   size of outputData
  * @return ccStatus_t
  */
ccStatus_t ccTransFilter(const ccFilterDescriptor_t wDesc,
                         const void *w,
                         const ccFilterDescriptor_t yDesc,
                         void *y,
                         uint32_t ySizeInBytes);

/**
  * @ingroup dnn
  * @brief trans weight to fractal format, and trans data type together
  * @param [in] wDesc   descriptor of input filter
  * @param [in] w   input data pointer
  * @param [in] dataTypeTransmode   mode of data type transform
  * @param [in] yDesc   descriptor of output filter
  * @param [in|out] y   output data pointer
  * @param [in] ySizeInBytes   size of outputData
  * @return ccStatus_t
  */
ccStatus_t ccTransFilterInt8(const ccFilterDescriptor_t wDesc,
                         const void *w,
                         ccFilterDescriptor_t yDesc,
                         void *y,
                         uint32_t ySizeInBytes,
                         ccDataType_t outputDataType);

/**
  * @ingroup dnn
  * @brief create descriptor of filter
  * @param [in|out] filterDesc   point to descriptor of filter
  * @return ccStatus_t
  */
ccStatus_t ccCreateFilterDescriptor(ccFilterDescriptor_t *filterDesc);

/**
  * @ingroup dnn
  * @brief destroy descriptor of filter
  * @param [in] *filterDesc   descriptor of filter
  * @return ccStatus_t
  */
ccStatus_t ccDestroyFilterDescriptor(ccFilterDescriptor_t *filterDesc);

/**
  * @ingroup dnn
  * @brief init conv descriptor to 2d conv
  * @param [in|out] convDesc   descriptor of convolution operator
  * @param [in] mode   mode of convolution
  * @param [in] padMode   mode of padding
  * @param [in] padHHead   zero padding in height head, if padMode is not CC_PADDING_DIRECTASSIGN head and tail is same value
  * @param [in] padHTail   zero padding in height tail, need set when padMode is CC_PADDING_DIRECTASSIGN.
  * @param [in] padWHead   zero padding in width head,  if padMode is not CC_PADDING_DIRECTASSIGN head and tail is same value
  * @param [in] padWTail   zero padding in width tail, need set when padMode is CC_PADDING_DIRECTASSIGN
  * @param [in] strideH   stride in height
  * @param [in] strideW   stride in width
  * @param [in] dilationH   dilation in height
  * @param [in] dilationW   dilation in width
  * @return ccStatus_t
  */
ccStatus_t ccSetConvolution2dDescriptor(ccConvolutionDescriptor_t convDesc,
                                        ccConvolutionMode_t mode,
                                        ccPaddingMode_t padMode,
                                        int32_t padHHead,
                                        int32_t padHTail,
                                        int32_t padWHead,
                                        int32_t padWTail,
                                        int32_t group,
                                        int32_t strideH,
                                        int32_t strideW,
                                        int32_t dilationH,
                                        int32_t dilationW);

/**
  * @ingroup dnn
  * @brief read 2d conv
  * @param [in] convDesc   descriptor of convolution operator
  * @param [in|out] mode   point to mode of convolution
  * @param [in] padMode   mode of padding
  * @param [in] padHHead   zero padding in height head, if padMode is not CC_PADDING_DIRECTASSIGN head and tail is same value
  * @param [in] padHTail   zero padding in height tail, need set when padMode is CC_PADDING_DIRECTASSIGN.
  * @param [in] padWHead   zero padding in width head,  if padMode is not CC_PADDING_DIRECTASSIGN head and tail is same value
  * @param [in] padWTail   zero padding in width tail, need set when padMode is CC_PADDING_DIRECTASSIGN
  * @param [in|out] strideH   point to stride in height
  * @param [in|out] strideW   point to stride in width
  * @param [in|out] dilationH   point to dilation in height
  * @param [in|out] dilationW   point to dilation in width
  * @return ccStatus_t
  */
ccStatus_t ccGetConvolution2dDescriptor(const ccConvolutionDescriptor_t convDesc,
                                        ccConvolutionMode_t *mode,
                                        ccPaddingMode_t *padMode,
                                        int32_t *padHHead,
                                        int32_t *padHTail,
                                        int32_t *padWHead,
                                        int32_t *padWTail,
                                        int32_t *group,
                                        int32_t *strideH,
                                        int32_t *strideW,
                                        int32_t *dilationH,
                                        int32_t *dilationW);

/**
  * @ingroup dnn
  * @brief get the output dimension info of 2d convolution
  * @param [in] convDesc   descriptor of convolution operator
  * @param [in] xDesc   descriptor of input tensor
  * @param [in] wDesc   descriptor of filter
  * @param [in|out] n   point to batch size
  * @param [in|out] c   point to channels
  * @param [in|out] h   point to height of feature map
  * @param [in|out] w   point to width of feature map
  * @return ccStatus_t
  */
ccStatus_t ccGetConvolution2dForwardOutputDim(const ccConvolutionDescriptor_t  convDesc,
                                              const ccTensorDescriptor_t xDesc,
                                              const ccFilterDescriptor_t wDesc,
                                              int32_t *n,
                                              int32_t *c,
                                              int32_t *h,
                                              int32_t *w);

/**
  * @ingroup dnn
  * @brief create descriptor of convolution operator
  * @param [in|out] filterDesc   point to descriptor of convolution operator
  * @return ccStatus_t
  */
ccStatus_t ccCreateConvolutionDescriptor(ccConvolutionDescriptor_t *convDesc);

/**
  * @ingroup dnn
  * @brief destroy descriptor of convolution operator
  * @param [in] *convDesc   descriptor of convolution operator
  * @return ccStatus_t
  */
ccStatus_t ccDestroyConvolutionDescriptor(ccConvolutionDescriptor_t *convDesc);

/**
  * @ingroup dnn
  * @brief check specific stride condition flag
  * @param [in] deconvDesc   descriptor of Deconvolution operator
  * @param [in] xDesc   descriptor of input tensor
  * @param [in] yDesc   descriptor of output tensor
  * @param [in] biasDesc   descriptor of bias tensor
  * @param [in] wDesc   descriptor of filter
  * @param [in|out] transMark   output condition flag
  * @return ccStatus_t
  */
ccStatus_t ccDeconvSpStrideCondCheck(const ccConvolutionDescriptor_t deconvDesc,
                                     const ccTensorDescriptor_t xDesc,
                                     const ccTensorDescriptor_t yDesc,
                                     const ccTensorDescriptor_t biasDesc,
                                     const ccFilterDescriptor_t wDesc,
                                     uint32_t &transMark);

/**
  * @ingroup dnn
  * @brief special deconv stride trans
  * @param [in] deconvDesc   descriptor of Deconvolution operator
  * @param [in] xDesc   descriptor of input tensor
  * @param [in] yDesc   descriptor of output tensor
  * @param [in] biasDesc   descriptor of bias tensor
  * @param [in] deconvStPtr   descriptor of filter
  * @param [in|out] xStPtr   descriptor of trans input tensor
  * @param [in|out] yStPtr   descriptor of trans output tensor
  * @param [in|out] wStPtr   descriptor of trans filter tensor
  * @param [in|out] wDesc   descriptor of trasn filter
  * @param [in|out] transMark   condition flag
  * @return ccStatus_t
  */
ccStatus_t ccDeconvSpStrideDescTrans(const ccConvolutionDescriptor_t deconvDesc,
                          const ccTensorDescriptor_t xDesc,
                          const ccTensorDescriptor_t yDesc,
                          const ccTensorDescriptor_t biasDesc __attribute__((__unused__)),
                          const ccFilterDescriptor_t wDesc,
                          ccConvolutionDescriptor_t deconvStPtr,
                          ccTensorDescriptor_t xStPtr,
                          ccTensorDescriptor_t yStPtr,
                          ccFilterDescriptor_t wStPtr,
                          uint32_t transMark);

/**
  * @ingroup dnn
  * @brief check deconv goto aicore flag
  * @param [in] deconvDesc   descriptor of Deconvolution operator
  * @param [in] xDesc   descriptor of input tensor
  * @param [in] yDesc   descriptor of output tensor
  * @param [in] wDesc   descriptor of filter
  * @param [in] isGotoAicore  out flag
  * @param [in] transMark   condition flag
  * @return ccStatus_t
  */
ccStatus_t ccDeconvCheckGotoAiCore(const ccConvolutionDescriptor_t deconvDesc,
                           const ccTensorDescriptor_t xDesc,
                           const ccTensorDescriptor_t yDesc,
                           const ccFilterDescriptor_t wDesc,
                           uint32_t *isGotoAicore,
                           uint32_t transMark);

/**
  * @ingroup dnn
  * @brief get the output dimension info of 2d Deconvolution
  * @param [in] deconvDesc   descriptor of Deconvolution operator
  * @param [in] xDesc   descriptor of input tensor
  * @param [in] wDesc   descriptor of filter
  * @param [in|out] n   point to batch size
  * @param [in|out] c   point to channels
  * @param [in|out] h   point to height of feature map
  * @param [in|out] w   point to width of feature map
  * @return ccStatus_t
  */
ccStatus_t ccGetDeconvolution2dForwardOutputDim(const ccConvolutionDescriptor_t deconvDesc,
                                                const ccTensorDescriptor_t xDesc,
                                                const ccFilterDescriptor_t wDesc,
                                                int32_t *n,
                                                int32_t *c,
                                                int32_t *h,
                                                int32_t *w);

/**
  * @ingroup dnn
  * @brief create descriptor of PAD
  * @param [in|out] padDesc  point to descriptor of pad
  * @return ccStatus_t
  */
ccStatus_t ccCreatePadDescriptor(ccPadDescriptor_t *padDesc);

/**
  * @ingroup dnn
  * @brief destroy descriptor of PAD
  * @param [in] *padDesc descriptor of PAD
  * @return ccStatus_t
  */
ccStatus_t ccDestroyPadDescriptor(ccPadDescriptor_t *padDesc);

/**
  * @ingroup dnn
  * @brief set PADDesc
  * @param [in|out] padDesc descriptor of PAD
  * @param [in] padMode  mode of PAD
  * @param [in] padValue  pad value of PAD
  * @param [in] wleft width left pad of PAD
  * @param [in] wright width right of PAD
  * @param [in] htop higth pad of PAD
  * @param [in] hbottom higth bottom pad of PAD
  * @return ccStatus_t
  */
ccStatus_t ccSetPadDescriptor(ccPadDescriptor_t padDesc,
                                    ccPadMode_t  padMode,
                                    float padValue,
                                    int32_t htop,
                                    int32_t hbottom,
                                    int32_t wleft,
                                    int32_t wright);

/**
  * @ingroup dnn
  * @brief read 2d pooling
  * @param [in] poolingDesc   descriptor of pooling operator
  * @param [in|out] mode   point to mode of pooling
  * @param [in|out] maxpoolingNanOpt   point to Nan propagation mode
  * @param [in|out] windowH   point to height of pooling window
  * @param [in|out] windowW   point to width of pooling window
  * @param [in|out] padHHead   point to zero padding in height head, if padMode is not CC_PADDING_DIRECTASSIGN head and tail is same value.
  * @param [in|out] padHTail   point to zero padding in height tail, need set when padMode is CC_PADDING_DIRECTASSIGN.
  * @param [in|out] padWHead   point to zero padding in width head, if padMode is not CC_PADDING_DIRECTASSIGN head and tail is same value.
  * @param [in|out] padWTail   point to zero padding in width tail, need set when padMode is CC_PADDING_DIRECTASSIGN.
  * @param [in|out] strideH   point to stride in height
  * @param [in|out] strideW   point to stride in width
  * @param [in|out] dataMode
  * @param [in|out] ceilMode  0:Ceil 1:Floor
  * @return ccStatus_t
  */
ccStatus_t ccGetPooling2dDescriptor(const ccPoolingDescriptor_t poolingDesc,
                                    ccPoolingMode_t *mode,
                                    ccPaddingMode_t *padMode,
                                    ccNanPropagation_t *maxpoolingNanOpt,
                                    int32_t *windowH,
                                    int32_t *windowW,
                                    int32_t *padHHead,
                                    int32_t *padHTail,
                                    int32_t *padWHead,
                                    int32_t *padWTail,
                                    int32_t *strideH,
                                    int32_t *strideW,
                                    int32_t *dataMode,
                                    int32_t *ceilMode,
                                    ccPooingFwdAlgo_t *algo);

ccStatus_t ccGetCompare5dOutputDim(const ccTensorDescriptor_t xDesc,
                                   const ccTensorDescriptor_t yDesc,
                                   int32_t* dimCnt,
                                   int32_t* dim,
                                   int32_t dimLen);

ccStatus_t ccGetMaximum5dOutputDim(const ccTensorDescriptor_t xDesc,
                                   const ccTensorDescriptor_t yDesc,
                                   int32_t* dimCnt,
                                   int32_t* dim,
                                   int32_t dimLen);

ccStatus_t ccGetMinimum5dOutputDim(const ccTensorDescriptor_t xDesc,
                                   const ccTensorDescriptor_t yDesc,
                                   int32_t* dimCnt,
                                   int32_t* dim,
                                   int32_t dimLen);

ccStatus_t ccGetReduce5dOutputDim(const ccTensorDescriptor_t xDesc,
                                const ccIntArray_t* axis,
                                bool keepDims,
                                int32_t *dimCnt,
                                int32_t dim[],
                                int32_t dimLen);

/**
 * @brief get out put descrition of slice tensor.
 * @param [in] xDesc         descriptor of input data
 * @param [in] begin         begin position of tensor
 * @param [in] size          size to slice
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
 ccStatus_t ccGetSliceOutputDim(
      const ccTensorDescriptor_t xDesc,
      const ccIntArray_t* begin,
      const ccIntArray_t* size,
      int32_t *dimCnt,
      int32_t dim[],
      int32_t dimLen);

/**
 * @ingroup dnn
 * @brief get strided slice output dim info.
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] stridedSliceDesc specifies the begin, end, strides of slice
 * @param [in] attrDesc         reserve for optional attributes.
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetStridedSliceOutputDim(const ccTensorDescriptor_t xDesc,
                                    const ccStridedSliceDescriptor_t stridedSliceDesc,
                                    const ccStridedSliceAttrsDescriptor_t attrDesc,
                                    int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
  * @ingroup dnn
  * @brief get workspace size for softmax computation
  * @param [in] handle              cce handle
  * @param [in] xDesc               descriptor of input tensor
  * @param [in] yDesc               descriptor of output tensor
  * @param [in|out] sizeInBytes     workSpace size in bytes
  * @return ccStatus_t
  */
ccStatus_t ccGetSoftmaxForwardWorkspaceSize(ccHandle_t handle,
                                            const ccTensorDescriptor_t xDesc,
                                            const ccTensorDescriptor_t yDesc,
                                            uint32_t *sizeInBytes);

/**
  * @ingroup dnn
  * @brief set quantize algorithm type and quantize scale type (vector or scalar)
  * @param [in] quantizeInfo    descriptor of quantize parameters
  * @param [in] quantAlgo       enum type for quantize algorithm type
  * @param [in] scaleType       enum type for quantize scale type
  * @param [in] reluflag        flag for relu
  * @return ccStatus_t
  */
ccStatus_t   ccSetQuantizeAlgoAndScaleType(ccQuantizeDescriptor_t quantizeInfo, ccQuantizeAlgo_t quantAlgo, ccScaleType_t scaleType);
ccStatus_t   ccSetQuantizeAlgoAndScaleType(ccQuantizeDescriptor_t quantizeInfo, ccQuantizeAlgo_t quantAlgo, ccScaleType_t scaleType, bool reluFlag);

}; /* end cce */

#endif  // DNN_BASE_H__

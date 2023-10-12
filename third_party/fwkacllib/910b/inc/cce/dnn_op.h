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

#ifndef DNN_OP_H__
#define DNN_OP_H__

#include "cce/blas_struct.h"
#include "cce/cce.h"
#include "cce/customize.h"

namespace cce {

/**
 * @ingroup dnn
 * @brief create descriptor of parameters for exponential function
 * @param [in] point to descriptor of parameters for exponential function
 * @return ccStatus_t
 */
ccStatus_t ccCreateExpDescriptor(ccExpDescriptor_t *expDesc);

/**
 * @ingroup dnn
 * @brief create descriptor of parameters for logarithmic function
 * @param [in] point to descriptor of parameters for logarithmic function
 * @return ccStatus_t
 */

ccStatus_t ccCreateLogDescriptor(ccLogDescriptor_t *logDesc);

/**
 * @ingroup dnn
 * @brief create descriptor of parameters for pow function
 * @param [in] point to descriptor of parameters for pow function
 * @return ccStatus_t
 */

ccStatus_t ccCreatePowDescriptor(ccPowDescriptor_t *powDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of parameters for exponential function
 * @param [in] point to descriptor of parameters for exponential function
 * @return ccStatus_t
 */
ccStatus_t ccDestroyExpDescriptor(ccExpDescriptor_t *expDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of parameters for logarithmic function
 * @param [in] point to descriptor of parameters for exponential function
 * @return ccStatus_t
 */
ccStatus_t ccDestroyLogDescriptor(ccLogDescriptor_t *logDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of parameters for pow function
 * @param [in] point to descriptor of parameters for pow function
 * @return ccStatus_t
 */
ccStatus_t ccDestroyPowDescriptor(ccPowDescriptor_t *powDesc);

/**
 * @ingroup dnn
 * @brief create descriptor of parameters for NonMaxSuppress function
 * @param [in] point to descriptor of parameters for NonMaxSuppress function
 * @return ccStatus_t
 */
ccStatus_t ccCreateNonMaxSuppressionDescriptor(ccNonMaxSuppressionDescriptor_t *nonMaxSuppressionDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of parameters for NonMaxSuppress function
 * @param [in] point to descriptor of parameters for NonMaxSuppress function
 * @return ccStatus_t
 */
ccStatus_t ccDestroyNonMaxSuppressionDescriptor(ccNonMaxSuppressionDescriptor_t *nonMaxSuppressionDesc);

ccStatus_t ccTransTensorIncertPads(const ccTensorDescriptor_t xDesc, const void *x, const ccTensorDescriptor_t yDesc,
                                   void *y, uint32_t ySizeInBytes, uint32_t boxTypeNum, bool interweave,
                                   bool background, uint32_t boxTypeNumMax = 0, bool isScaleVec = false);

ccStatus_t ccTransTensorIncertPadsInt32(const ccTensorDescriptor_t xDesc, const void *x,
                                        const ccTensorDescriptor_t yDesc, void *y, uint32_t ySizeInBytes,
                                        uint32_t boxTypeNum, bool interweave, bool background);

ccStatus_t ccTransMskrcnnBbox(const ccTensorDescriptor_t xDesc, const void *x, const ccTensorDescriptor_t yDesc,
                              void *y, uint32_t ySizeInBytes, uint32_t boxTypeNum);

ccStatus_t ccSetTensorDescriptorQuantizeParam(ccTensorDescriptor_t tensorDesc,
                                              const ccVecQuantizePara_t *vecQuantizePara);

ccStatus_t ccGetTensorDescriptorQuantizeParam(const ccTensorDescriptor_t tensorDesc,
                                              ccVecQuantizePara_t *vecQuantizePara);

/**
 * @ingroup dnn
 * @brief init tensor to 4d filter
 * @param [in|out] filterDesc   descriptor of filter
 * @param [in] format   format of filter
 * @param [in] dataType   data type in device
 * @param [in] k   number of output feature maps
 * @param [in] c   number of input feature maps
 * @param [in] h   height of filter
 * @param [in] w   width of filter
 * @return ccStatus_t
 */
ccStatus_t ccSetFilter4dDescriptor(ccFilterDescriptor_t filterDesc, ccTensorFormat_t format, ccDataType_t dataType,
                                   int32_t k, int32_t c, int32_t h, int32_t w);

ccStatus_t ccSetFilter6dDescriptor(ccTensorDescriptor_t filterDesc, ccTensorFormat_t format, ccDataType_t dataType,
                                   int32_t c1, int32_t h, int32_t w, int32_t n, int32_t co, int32_t c0);
/**
 * @ingroup dnn
 * @brief init tensor to Fractal filter
 * @param [in|out] filterDesc   descriptor of filter
 * @param [in] format   format of filter
 * @param [in] dataType   data type in device
 * @param [in] k   number of output feature maps
 * @param [in] c   number of input feature maps
 * @param [in] h   height of filter
 * @param [in] w   width of filter
 * @return ccStatus_t
 */
ccStatus_t ccSetFilterFractalDescriptor(ccFilterDescriptor_t filterDesc, ccTensorFormat_t format, ccDataType_t dataType,
                                        int32_t k, int32_t c, int32_t h, int32_t w);

/**
 * @ingroup dnn
 * @brief init tensor to Fractal filter
 * @param [in|out] filterDesc   descriptor of filter
 * @param [in] format   format of filter
 * @param [in] dataType   data type in device
 * @param [in] k   number of output feature maps
 * @param [in] c   number of input feature maps
 * @param [in] h   height of filter
 * @param [in] w   width of filter
 * @return ccStatus_t
 */
ccStatus_t ccSetInt8Filter4dDescriptor(ccFilterDescriptor_t filterDesc, ccTensorFormat_t format, ccDataType_t dataType,
                                       int32_t k, int32_t c, int32_t h, int32_t w, ccDataType_t outputDataType);

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
ccStatus_t ccGetFilterFractalDescriptor(const ccFilterDescriptor_t filterDesc, ccTensorFormat_t *format,
                                        ccDataType_t *dataType, int32_t *k, int32_t *c, int32_t *h, int32_t *w);

/**
 * @ingroup dnn
 * @brief get data size of 4d filter
 * @param [in] filterDesc   descriptor of filter
 * @param [in|out] size   point to data size
 * @return ccStatus_t
 */
ccStatus_t ccGetDepthWiseConvFilterSizeInBytes(const ccFilterDescriptor_t filterDesc, int32_t groupNum, uint32_t *size);

/**
 * @ingroup dnn
 * @brief trans group conv filter to fractal format
 * @param [in] filterSrcInfo   descriptor of input filter
 * @param [in] filterSrc   input data pointer
 * @param [in] filterDstInfo   descriptor of output filter
 * @param [in|out] filterDst   output data pointer
 * @param [in] group   group size
 * @return ccStatus_t
 */
ccStatus_t ccTransGroupConvFilter(ccFilterDescriptor_t filterSrcInfo, const void *filterSrc,
                                  ccFilterDescriptor_t filterDstInfo, void *filterDst, uint32_t group,
                                  uint32_t dstSize);

/**
 * @ingroup dnn
 * @brief trans conv filter With BoxTypeNuM to fractal format
 * @param [in] filterSrcInfo   descriptor of input filter
 * @param [in] filterSrc   input data pointer
 * @param [in] filterDstInfo   descriptor of output filter
 * @param [in|out] filterDst   output data pointer
 * @param [in] ySizeInBytes the malloc memory size
 * @param [in] boxTypeNum  the num of boxType
 * @param [in] interweave whether the axis interweave
 * @return ccStatus_t
 */
ccStatus_t ccTransFilterWithBoxTypeNum(const ccFilterDescriptor_t xDesc, const void *x,
                                       const ccFilterDescriptor_t yDesc, void *y, uint32_t ySizeInBytes,
                                       uint32_t boxTypeNum, bool interweave, uint32_t boxTypeNumMax = 0);
/**
 * @ingroup dnn
 * @brief trans conv filter With BoxTypeNuM to fractal format
 * @param [in] filterSrcInfo   descriptor of input filter
 * @param [in] filterSrc   input data pointer
 * @param [in] filterDstInfo   descriptor of output filter
 * @param [in|out] filterDst   output data pointer
 * @param [in] ySizeInBytes the malloc memory size
 * @param [in] boxTypeNum  the num of boxType
 * @param [in] interweave whether the axis interweave
 * @param [in] outputDataType  output DataType
 * @return ccStatus_t
 */
ccStatus_t ccTransFilterInt8WithBoxTypeNum(const ccFilterDescriptor_t wDesc, const void *x,
                                           const ccFilterDescriptor_t yDesc, void *y, uint32_t ySizeInBytes,
                                           uint32_t boxTypeNum, bool interweave, ccDataType_t outputDataType);

/**
 * @ingroup dnn
 * @brief trans depthwise conv filter  to fractal format
 * @param [in]  wDesc descriptor of input filter
 * @param [in] w   input data pointer
 * @param [in] groupNum   groupNum of conv
 * @param [in]..yDesc descriptor of output filter
 * @param [in|out] y   output data pointer
 * @param [in] ySizeInBytes the malloc memory size
 * @return ccStatus_t
 */

ccStatus_t transDepthWiseConvFilterNCHWToFractalZ(const ccFilterDescriptor_t wDesc, const void *w, int32_t groupNum,
                                                  ccFilterDescriptor_t yDesc, void *y, uint32_t ySizeInBytes);

/**
 * @ingroup dnn
 * @brief trans depthwise conv filter  to fractal format
 * @param [in]  wDesc descriptor of input filter
 * @param [in] w   input data pointer
 * @param [in] groupNum   groupNum of conv
 * @param [in]..yDesc descriptor of output filter
 * @param [in|out] y   output data pointer
 * @param [in] ySizeInBytes the malloc memory size
 * @return ccStatus_t
 */
ccStatus_t transDepthWiseConvFilterInt8NCHWToFractalZ(const ccFilterDescriptor_t wDesc, const void *w, int32_t groupNum,
                                                      ccFilterDescriptor_t yDesc, void *y, uint32_t ySizeInBytes);

/**
 * @ingroup dnn
 * @brief trans depthwise conv filter  to fractal format, input format CHWN
 * @param [in]  wDesc descriptor of input filter
 * @param [in]..yDesc descriptor of output filter
 * @param [in] ySizeInBytes the malloc memory size
 * @param [in] w   input data pointer
 * @param [in|out] y   output data pointer
 * @return ccStatus_t
 */
ccStatus_t transDepthWiseConvFilterCHWNToFractalZ(ccFilterDescriptor_t filterSrcInfo, const void *filterSrc,
                                                  uint32_t group, ccFilterDescriptor_t filterDstInfo, void *filterDst,
                                                  uint32_t destSize);

/**
 * @ingroup dnn
 * @Check if it is surpported by HighPerformance depthwise
 * @param [in]  inputN,C,H,W   input param
 * @param [in]  filterN,C,H,W
 * @param [in]  dilationH,W    dilation param
 * @param [in]  padHHead,padHtail,padWHead,padWTail pad param
 * @param [in]  strideH,W  stride param
 * @param [in] groupNum  Conv groupNum
 * @param [in|out] isHighPerformance  isHighPerformance flag
 * @return ccStatus_t
 */
ccStatus_t ccIsDepthwiseHighPerformance(int32_t inputN, int32_t inputC, int32_t inputH, int32_t inputW, int32_t filterN,
                                        int32_t filterC, int32_t filterH, int32_t filterW, int32_t dilationH,
                                        int32_t dilationW, int32_t padHHead, int32_t padHTail, int32_t padWHead,
                                        int32_t padWTail, int32_t strideH, int32_t strideW, int32_t groupNum,
                                        bool &isHighPerformance, bool isquant = false,
                                        ccDataType_t inputDataType = CC_DATA_HALF,
                                        ccDataType_t outputDataType = CC_DATA_HALF);

/**
 * @ingroup dnn
 * @brief trans depthwise conv filter  to fractal format, input format CHWN
 * @param [in]  wDesc descriptor of input filter
 * @param [in]..yDesc descriptor of output filter
 * @param [in] ySizeInBytes the malloc memory size
 * @param [in] w   input data pointer
 * @param [in|out] y   output data pointer
 * @return ccStatus_t
 */
ccStatus_t transDepthWiseConvFilterCHWNToFractalZ(ccFilterDescriptor_t filterSrcInfo, const void *filterSrc,
                                                  uint32_t group, ccFilterDescriptor_t filterDstInfo, void *filterDst,
                                                  uint32_t destSize);

/**
 * @ingroup dnn
 * @brief create descriptor of fullconnection operator
 * @param [in|out] fcDesc   point to descriptor of fullconnection operator
 * @return ccStatus_t
 */
ccStatus_t ccCreateFullConnectionDescriptor(ccFullConnectionDescriptor_t *fcDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of fullconnection operator
 * @param [in] *fcDesc   descriptor of fullconnection operator
 * @return ccStatus_t
 */
ccStatus_t ccDestroyFullConnectionDescriptor(ccFullConnectionDescriptor_t *fcDesc);

/**
 * @ingroup dnn
 * @brief init conv descriptor to 2d conv, use for beforeHasPad
 * @param [in|out] convDesc   descriptor of convolution operator
 * @param [in] beforepadHHead   before padding in height head
 * @param [in] beforepadHTail   before padding in height tail
 * @param [in] beforepadWHead   before padding in width head
 * @param [in] beforepadWTail   before padding in width tail
 * @return ccStatus_t
 */
ccStatus_t ccSetConvolution2dDescriptorForPad(ccConvolutionDescriptor_t convDesc, int32_t beforepadHHead,
                                              int32_t beforepadHTail, int32_t beforepadWHead, int32_t beforepadWTail);

/**
 * @ingroup dnn
 * @brief init conv descriptor to 2d conv, use for concat batch size
 * @param [in|out] convDesc   descriptor of convolution operator
 * @param [in] concatBatchSize   concat batch size
 * @return ccStatus_t
 */
ccStatus_t ccSetConvolution2dDescriptorForConcatBatchSize(ccConvolutionDescriptor_t convDesc, int64_t concatBatchSize);

/**
 * @ingroup dnn
 * @brief init conv descriptor to 2d conv
 * @param [in|out] convDesc   descriptor of convolution operator
 * @param [in] opType  operation type for append at convolution operation
 * @param [in] opDesc  operation descritpor for the opType
 * @return ccStatus_t
 */
ccStatus_t ccConvolution2dAppendOp(ccConvolutionDescriptor_t convDesc, ccOpType_t opType, const void *opDesc);

/**
 * @ingroup dnn
 * @brief read 2d conv beforeHasPad
 * @param [in] convDesc   descriptor of convolution operator
 * @param [in|out] beforepadHHead   before padding in height head, default is 0
 * @param [in|out] beforepadHTail   before padding in height tail, default is 0
 * @param [in|out] beforepadWHead   before padding in width head, default is 0
 * @param [in|out] beforepadWTail   before padding in width tail, default is 0
 */
ccStatus_t ccGetConvolution2dDescriptorForPad(const ccConvolutionDescriptor_t convDesc, int32_t *beforepadHHead,
                                              int32_t *beforepadHTail, int32_t *beforepadWHead,
                                              int32_t *beforepadWTail);

/**
 * @ingroup dnn
 * @brief read 2d conv concat batch size
 * @param [in] convDesc   descriptor of convolution operator
 * @param [in|out] concatBatchSize   concat batch size, default is 0
 */
ccStatus_t ccGetConvolution2dDescriptorForConcatBatchSize(const ccConvolutionDescriptor_t convDesc,
                                                          int64_t *concatBatchSize);

/**
 * @ingroup dnn
 * @brief get the temp space size of convolution forward computation, maybe no need temp space
 * @param [in] handle   cce handle
 * @param [in] convDesc   descriptor of convolution operator
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] wDesc   descriptor of filter
 * @param [in] yDesc   descriptor of output tensor
 * @param [in] algo   algorithm of convolution forward
 * @param [in|out] sizeInBytes   temp space size need for specified algorithm
 * @return ccStatus_t
 */
ccStatus_t ccGetConvolutionForwardWorkspaceSize(ccHandle_t handle, const ccConvolutionDescriptor_t convDesc,
                                                const ccTensorDescriptor_t xDesc, const ccFilterDescriptor_t wDesc,
                                                const ccTensorDescriptor_t yDesc, ccConvolutionFwdAlgo_t algo,
                                                uint32_t *sizeInBytes);
/**
 * @ingroup dnn
 * @brief get the temp space size of convolution backward computation, maybe no need temp space
 * @param [in] handle   cce handle
 * @param [in] convDesc   descriptor of convolution operator
 * @param [in] dyDesc   descriptor of input tensor
 * @param [in] wDesc   descriptor of filter
 * @param [in] dxDesc   descriptor of output tensor
 * @param [in] algo   algorithm of convolution forward
 * @param [in|out] sizeInBytes   temp space size need for specified algorithm
 * @return ccStatus_t
 */
ccStatus_t ccGetConvolutionBackwardDataWorkspaceSize(ccHandle_t handle, const ccConvolutionDescriptor_t convDesc,
                                                     const ccTensorDescriptor_t dyDesc,
                                                     const ccFilterDescriptor_t wDesc,
                                                     const ccTensorDescriptor_t dxDesc, ccConvolutionBwdAlgo_t algo,
                                                     uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief get the temp space size of fc forward computation, maybe no need temp space
 * @param [in] handle  cce handle
 * @param [in] fcDesc  descriptor of fc operator
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] wDesc   descriptor of filter
 * @param [in] yDesc   descriptor of output tensor
 * @param [in|out] sizeInBytes   temp space size need, 0 means no memeory needed
 * @return ccStatus_t
 */
ccStatus_t ccGetFullConnectionForwardWorkspaceSize(ccHandle_t handle, const ccFullConnectionDescriptor_t fcDesc,
                                                   const ccTensorDescriptor_t xDesc, const ccFilterDescriptor_t wDesc,
                                                   const ccTensorDescriptor_t yDesc, uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief convolution forward computation
 * @param [in] handle   cce handle
 * @param [in] convDesc   descriptor of convolution operator
 * @param [in] alpha   scaling factors
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] x   input data in device memory
 * @param [in] wDesc   descriptor of filter
 * @param [in] w   filter data in device memory
 * @param [in] biasDesc   descriptor of bias
 * @param [in] bias   bias data in device memory
 * @param [in] algo   algorithm of convolution forward
 * @param [in] workSpace   temp space, maybe NULL if no need temp space
 * @param [in] workSpaceSizeInBytes   sizeof workspace
 * @param [in] beta   scaling factors
 * @param [in] yDesc   descriptor of output tensor
 * @param [in|out] y   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccConvolutionForward(ccHandle_t handle, const ccConvolutionDescriptor_t convDesc, const void *alpha,
                                const ccTensorDescriptor_t xDesc, const void *x, const ccFilterDescriptor_t wDesc,
                                const void *w, const ccTensorDescriptor_t biasDesc, const void *bias,
                                ccConvolutionFwdAlgo_t algo, void *workSpace, uint32_t workSpaceSizeInBytes,
                                const void *beta, const ccTensorDescriptor_t yDesc, void *y);
/**
 * @ingroup dnn
 * @brief full alloc float and reset to 0
 * @param [in] handle      cce handle
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in|out] x       output data in device memory
 * @param [in] beta        scaling factors
 * @return ccStatus_t
 */
ccStatus_t ccAllocFloatStatus(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                              const void *beta);

/**
 * @ingroup dnn
 * @brief full get data set by op
 * @param [in] handle      cce handle
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in|out] x       output data in device memory
 * @param [in] beta        scaling factors
 * @param [in] yDesc       descriptor of output tensor
 * @param [out] y          output data in device memory
 * @return ccStatus_t
 */

ccStatus_t ccGetFloatStatus(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                            const void *beta, const ccTensorDescriptor_t yDesc, const void *y);

/**
 * @ingroup dnn
 * @brief full clear register
 * @param [in] handle      cce handle
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] x           input data in device memory
 * @param [in] beta        scaling factors
 * @param [in] yDesc       descriptor of output tensor
 * @param [out] y          output data in device memory
 * @return ccStatus_t
 */

ccStatus_t ccClearFloatStatus(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                              const void *beta, const ccTensorDescriptor_t yDesc, const void *y);

#ifndef DAVINCI_LITE
/**
 * @ingroup dnn
 * @brief convolution backward data computation
 * @param [in] handle   cce handle
 * @param [in] convDesc   descriptor of convolution operator
 * @param [in] alpha   scaling factors
 * @param [in] dyDesc   descriptor of input tensor
 * @param [in] dy   input data in device memory
 * @param [in] wDesc   descriptor of filter
 * @param [in] w   filter data in device memory
 * @param [in] algo   algorithm of convolution backward
 * @param [in] workSpace   temp space, maybe NULL if no need temp space
 * @param [in] workSpaceSizeInBytes   sizeof workspace
 * @param [in] beta   scaling factors
 * @param [in] dxDesc   descriptor of output tensor
 * @param [in|out] dx   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccConvolutionBackwardData(ccHandle_t handle, const ccConvolutionDescriptor_t convDesc, const void *alpha,
                                     const ccTensorDescriptor_t dyDesc, const void *dy,
                                     const ccFilterDescriptor_t wDesc, const void *w, ccConvolutionBwdAlgo_t algo,
                                     void *workSpace, uint32_t workSpaceSizeInBytes, const void *beta,
                                     const ccTensorDescriptor_t dxDesc, void *dx);
#endif

/**
 * @ingroup dnn
 * @brief create descriptor of pooling operator
 * @param [in|out] poolingDesc   point to descriptor of pooling operator
 * @return ccStatus_t
 */
ccStatus_t ccCreatePoolingDescriptor(ccPoolingDescriptor_t *poolingDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of pooling operator
 * @param [in] *poolingDesc   descriptor of pooling operator
 * @return ccStatus_t
 */
ccStatus_t ccDestroyPoolingDescriptor(ccPoolingDescriptor_t *poolingDesc);

/**
 * @ingroup dnn
 * @brief init pooling descriptor to 2d pooling
 * @param [in|out] poolingDesc   descriptor of pooling operator
 * @param [in] mode   mode of pooling
 * @param [in] padMode   mode of padding
 * @param [in] maxpoolingNanOpt   Nan propagation mode
 * @param [in] windowH   height of pooling window
 * @param [in] windowW   width of pooling window
 * @param [in] padHHead   zero padding in height head, if padMode is not CC_PADDING_DIRECTASSIGN head and tail is same
 * value.
 * @param [in] padHTail   zero padding in height tail, need set when padMode is CC_PADDING_DIRECTASSIGN.
 * @param [in] padWHead   zero padding in width head,  if padMode is not CC_PADDING_DIRECTASSIGN head and tail is same
 * value.
 * @param [in] padWTail   zero padding in width tail, need set when padMode is CC_PADDING_DIRECTASSIGN..
 * @param [in] strideH   stride in height
 * @param [in] strideW   stride in width
 * @param [in] dataMode
 * @param [in] ceilMode   0:Floor  1:Ceil
 * @return ccStatus_t
 */
ccStatus_t ccSetPooling2dDescriptor(ccPoolingDescriptor_t poolingDesc, ccPoolingMode_t mode, ccPaddingMode_t padMode,
                                    ccNanPropagation_t maxpoolingNanOpt, int32_t windowH, int32_t windowW,
                                    int32_t padHHead, int32_t padHTail, int32_t padWHead, int32_t padWTail,
                                    int32_t strideH, int32_t strideW, int32_t dataMode, int32_t ceilMode,
                                    ccPooingFwdAlgo_t algo = CC_POOLING_FWD_ALGO_HALF);

/**
 * @ingroup dnn
 * @brief get the output dimension info of 2d pooling
 * @param [in] poolingDesc   descriptor of pooling operator
 * @param [in] xDesc   descriptor of input tensor
 * @param [in|out] n   point to batch size
 * @param [in|out] c   point to channels
 * @param [in|out] h   point to height of feature map
 * @param [in|out] w   point to width of feature map
 * @return ccStatus_t
 */
ccStatus_t ccGetPooling2dForwardOutputDim(const ccPoolingDescriptor_t poolingDesc, const ccTensorDescriptor_t xDesc,
                                          int32_t *n, int32_t *c, int32_t *h, int32_t *w);

/**
 * @ingroup dnn
 * @brief pooling forward computation
 * @param [in] handle   cce handle
 * @param [in] poolingDesc   descriptor of pooling operator
 * @param [in] alpha   scaling factors
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] x   input data in device memory
 * @param [in] beta   scaling factors
 * @param [in] yDesc   descriptor of output tensor
 * @param [in|out] y   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccPoolingForward(ccHandle_t handle, const ccPoolingDescriptor_t poolingDesc, const void *alpha,
                            const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                            const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief pooling backward computation
 * @param [in] handle   cce handle
 * @param [in] poolingDesc   descriptor of pooling operator
 * @param [in] alpha   scaling factors
 * @param [in] beta   scaling factors
 * @param [in] argMaskDesc   descriptor of mask tensor
 * @param [in] argMask   mask data in device memory
 * @param [in] dyDesc   descriptor of input tensor
 * @param [in] dy   input data in device memory
 * @param [in] dxDesc   descriptor of output tensor
 * @param [in|out] dx   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccMaxPoolingBackward(ccHandle_t handle, const ccPoolingDescriptor_t poolingDesc, const void *alpha,
                                const void *beta, const ccTensorDescriptor_t argMaskDesc, const void *argMask,
                                const ccTensorDescriptor_t dyDesc, const void *dy, const ccTensorDescriptor_t dxDesc,
                                void *dx);
/**
 * @ingroup dnn
 * @brief create descriptor of activation operator
 * @param [in|out] activationDesc   point to descriptor of activation operator
 * @return ccStatus_t
 */
ccStatus_t ccCreateActivationDescriptor(ccActivationDescriptor_t *activationDesc);

/**
 * @ingroup dnn
 * @brief init activation descriptor to 2d activation
 * @param [in|out] activationDesc   descriptor of activation operator
 * @param [in] mode                 mode of activation
 * @param [in] reluNanOpt           Nan propagation mode
 * @param [in] coef                 ceiling for clipped RELU, alpha for ELU
 * @param [in] activationPara       activation parameter union
 * @return ccStatus_t
 */
ccStatus_t ccSetActivationDescriptor(ccActivationDescriptor_t activationDesc, ccActivationMode_t mode,
                                     ccNanPropagation_t reluNanOpt, double coef,
                                     ccActivationPara_u activationPara = {{0, CC_NAN_NOT_PROPAGATE}});

/**
 * @ingroup dnn
 * @brief read activation param
 * @param [in] activationDesc     descriptor of activation operator
 * @param [in|out] mode           point to mode of activation
 * @param [in|out] reluNanOpt     point to Nan propagation mode
 * @param [in|out] coef           point to coef
 * @param [in|out] activationPara point to activation parameter union
 * @return ccStatus_t
 */
ccStatus_t ccGetActivationDescriptor(const ccActivationDescriptor_t activationDesc, ccActivationMode_t *mode,
                                     ccNanPropagation_t *reluNanOpt, double *coef,
                                     ccActivationPara_u *activationPara = NULL);

/**
 * @ingroup dnn
 * @brief destroy descriptor of activation operator
 * @param [in] *activationDesc   descriptor of activation operator
 * @return ccStatus_t
 */
ccStatus_t ccDestroyActivationDescriptor(ccActivationDescriptor_t *activationDesc);

/**
 * @ingroup dnn
 * @brief activation forward computation
 * @param [in] handle   cce handle
 * @param [in] activationDesc   descriptor of activation operator
 * @param [in] alpha   scaling factors
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] x   input data in device memory
 * @param [in] beta   scaling factors
 * @param [in] yDesc   descriptor of output tensor
 * @param [in|out] y   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccActivationForward(ccHandle_t handle, const ccActivationDescriptor_t activationDesc, const void *alpha,
                               const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                               const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief Derives a tensor descriptor from layer data descriptor for BatchNormalization
 * @param [in|out] derivedBnDesc   descriptor of mean, variance, bias, scale tensors tensor
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] mode    mode of BatchNormalization
 * @return ccStatus_t
 */
ccStatus_t ccDeriveBNTensorDescriptor(ccTensorDescriptor_t derivedBnDesc, const ccTensorDescriptor_t xDesc,
                                      ccBatchNormMode_t mode);

/**
 * @ingroup dnn
 * @brief batchnorm forward computation
 * @param [in] handle   cce handle
 * @param [in] mode     mode of batchnorm
 * @param [in] alpha    scaling factors
 * @param [in] beta     scaling factors
 * @param [in] xDesc    descriptor of input tensor
 * @param [in] x        input data in device memory
 * @param [in] yDesc    descriptor of output tensor
 * @param [in|out] y        output data in device memory
 * @param [in] bnScaleBiasMeanVarDesc  descriptor of scale, bias, mean, variance tensor
 * @param [in] bnScale       scaling factor
 * @param [in] bnBias        bias factor
 * @param [in] estimatedMean    mean
 * @param [in] estimatedVariance   variance
 * @param [in] epsilon     epsilon
 * @return ccStatus_t
 */
ccStatus_t ccBatchNormForwardInference(ccHandle_t handle, ccBatchNormMode_t mode, const void *alpha, const void *beta,
                                       const ccTensorDescriptor_t xDesc, const void *x,
                                       const ccTensorDescriptor_t yDesc, void *y,
                                       const ccTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
                                       const void *bnBias, const void *estimatedMean, const void *estimatedVariance,
                                       double epsilon);

/**
 * @ingroup dnn
 * @brief batchnorm forward computation
 * @param [in] handle   cce handle
 * @param [in] mode     mode of batchnorm
 * @param [in] reluFlag     relu fusion flag
 * @param [in] alpha    scaling factors
 * @param [in] beta     scaling factors
 * @param [in] xDesc    descriptor of input tensor
 * @param [in] x        input data in device memory
 * @param [in] yDesc    descriptor of output tensor
 * @param [in|out] y        output data in device memory
 * @param [in] bnScaleBiasMeanVarDesc  descriptor of scale, bias, mean, variance tensor
 * @param [in] bnScale       scaling factor
 * @param [in] bnBias        bias factor
 * @param [in] estimatedMean    mean
 * @param [in] estimatedVariance   variance
 * @param [in] epsilon     epsilon
 * @return ccStatus_t
 */
ccStatus_t ccBatchNormFusionForwardInference(ccHandle_t handle, ccBatchNormMode_t mode, ccBatchNormDescriptor_t bnDesc,
                                             const void *alpha, const void *beta, const ccTensorDescriptor_t xDesc,
                                             const void *x, const ccTensorDescriptor_t yDesc, void *y,
                                             const ccTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
                                             const void *bnBias, const void *estimatedMean,
                                             const void *estimatedVariance, double epsilon);

/**
 * @ingroup dnn
 * @brief create descriptor of batchnorm operator
 * @param [in|out] bnDesc   point to descriptor of batchnorm operator
 * @return ccStatus_t
 */
ccStatus_t ccCreateBatchNormDescriptor(ccBatchNormDescriptor_t *bnDesc);

/**
 * @ingroup dnn
 * @brief destroy batchnorm descriptor
 * @param [in] descriptor of batchnorm operator
 * @return ccStatus_t
 */
ccStatus_t ccDestroyBatchNormDescriptor(ccBatchNormDescriptor_t *bnDesc);

/**
 * @ingroup dnn
 * @brief append operation after batchnorm
 * @param [in|out] bnDesc   descriptor of batchnorm operator
 * @param [in] opType  operation type for append at batchnorm operation
 * @param [in] opDesc  operation descritpor for the opType
 * @return ccStatus_t
 */
ccStatus_t ccBatchNormAppendOp(ccBatchNormDescriptor_t bnDesc, ccOpType_t opType, const void *opDesc);

/**
 * @ingroup dnn
 * @brief full get the output 4d dimension info of full connection
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] wDesc       descriptor of weight tensor
 * @param [in|out] n       point to batch size
 * @param [in|out] c       point to channels
 * @param [in|out] h       point to height of feature map
 * @param [in|out] w       point to width of feature map
 * @return ccStatus_t
 */
ccStatus_t ccGetFullConnectionFwdOutputDim(const ccTensorDescriptor_t xDesc, const ccFilterDescriptor_t wDesc,
                                           int32_t *n, int32_t *c, int32_t *h, int32_t *w);

/**
 * @ingroup dnn
 * @brief full connection forward computation
 * @param [in] handle      cce handle
 * @param [in] fcDesc      fc desc
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] x           input data in device memory
 * @param [in] wDesc       descriptor of weight tensor
 * @param [in] w           filter data in device memory
 * @param [in] biasDesc    bias data in device memory
 * @param [in] bias        descriptor of bias tensor
 * @param [in] beta        scaling factors
 * @param [in] yDesc       descriptor of output tensor
 * @param [in|out] y       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccFullConnectionForwardEx2(ccHandle_t handle, const ccFullConnectionDescriptor_t fcDesc, const void *alpha,
                                      const ccTensorDescriptor_t xDesc, const void *x, const ccFilterDescriptor_t wDesc,
                                      const void *w, const ccTensorDescriptor_t biasDesc, const void *bias,
                                      const void *beta, const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief full connection forward computation with workspace
 * @param [in] handle      cce handle
 * @param [in] fcDesc      fc desc
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] x           input data in device memory
 * @param [in] wDesc       descriptor of weight tensor
 * @param [in] w           filter data in device memory
 * @param [in] biasDesc    bias data in device memory
 * @param [in] bias        descriptor of bias tensor
 * @param [in] workSpace   workSpace in device memory
 * @param [in] workSpaceSizeInBytes     workSpace size in bytes
 * @param [in] beta        scaling factors
 * @param [in] yDesc       descriptor of output tensor
 * @param [in|out] y       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccFullConnectionForwardWithWorkSpace(ccHandle_t handle, const ccFullConnectionDescriptor_t fcDesc,
                                                const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                                const ccFilterDescriptor_t wDesc, const void *w,
                                                const ccTensorDescriptor_t biasDesc, const void *bias, void *workSpace,
                                                uint32_t workSpaceSizeInBytes, const void *beta,
                                                const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief full softmax forward computation
 * @param [in] handle      cce handle
 * @param [in] algo        softmax algorithm
 * @param [in] mode        mode of softmax
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] x           input data in device memory
 * @param [in] workSpace   workSpace in device memory
 * @param [in] workSpaceSizeInBytes     workSpace size in bytes
 * @param [in] beta        scaling factors
 * @param [in] yDesc       descriptor of output tensor
 * @param [in|out] y       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccSoftmaxForward(ccHandle_t handle, ccSoftmaxAlgo_t algo, int32_t softmaxAxis, const void *alpha,
                            const ccTensorDescriptor_t xDesc, const void *x, void *workSpace,
                            uint32_t workSpaceSizeInBytes, const void *beta, const ccTensorDescriptor_t yDesc, void *y);
/**
 * @ingroup dnn
 * @brief full softmax forward computation
 * @param [in] handle      cce handle
 * @param [in] algo        softmax algorithm
 * @param [in] softmaxAxis mode of softmax
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] x           input data in device memory
 * @param [in] workSpace   workSpace in device memory
 * @param [in] workSpaceSizeInBytes     workSpace size in bytes
 * @param [in] beta        scaling factors
 * @param [in] yDesc       descriptor of output tensor
 * @param [in|out] y       output data in device memory
 * @param [in] classNum    class number
 * @param [in] padNum      pad Num
 * @return ccStatus_t
 */
ccStatus_t ccSoftmaxClassForward(ccHandle_t handle, ccSoftmaxAlgo_t algo, int32_t softmaxAxis, const void *alpha,
                                 const ccTensorDescriptor_t xDesc, const void *x, void *workSpace,
                                 uint32_t workSpaceSizeInBytes, const void *beta, const ccTensorDescriptor_t yDesc,
                                 void *y, uint32_t classNum, uint32_t padNum);

/**
 * @ingroup dnn
 * @brief full scale forward computation
 * @param [in] handle      cce handle
 * @param [in] scaleBiasDesc  descriptor of scale and bias tensor
 * @param [in] scale       scaling factor
 * @param [in] bias        bias factor
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] x           input data in device memory
 * @param [in] beta        scaling factors
 * @param [in] yDesc       descriptor of output tensor
 * @param [in|out] y       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccScaleForward(ccHandle_t handle, const ccTensorDescriptor_t scaleBiasDesc, const void *scale,
                          const void *bias, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          const void *beta, const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief full scale forward computation
 * @param [in] handle      cce handle
 * @param [in] scaleDesc  descriptor of scale and bias tensor
 * @param [in] scale       scaling factor
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] x           input data in device memory
 * @param [in] beta        scaling factors
 * @param [in] yDesc       descriptor of output tensor
 * @param [in|out] y       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccScaleNoBiasForward(ccHandle_t handle, const ccTensorDescriptor_t scaleDesc, const void *scale,
                                const void *alpha, const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief get the output dimension info of depth to space
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] blockSize        the size of block
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @return ccStatus_t
 */

ccStatus_t ccGetDepthToSpaceOutputDim(const ccTensorDescriptor_t xDesc, const int32_t blockSize, int32_t *dimCnt,
                                      int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief depth to space forward computation
 * @param [in] handle           cce handle
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] x                input data in device memory
 * @param [in] blockSize        the size of block
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */

ccStatus_t ccDepthToSpaceForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                 const int32_t blockSize, const void *beta, const ccTensorDescriptor_t outputDesc,
                                 void *output);

/**
 * @ingroup dnn
 * @brief get the output dimension info of space to depth
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] blockSize        the size of block
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @return ccStatus_t
 */

ccStatus_t ccGetSpaceToDepthOutputDim(const ccTensorDescriptor_t xDesc, const int32_t blockSize, int32_t *dimCnt,
                                      int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief space to depth forward computation
 * @param [in] handle           cce handle
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] x                input data in device memory
 * @param [in] blockSize        the size of block
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */

ccStatus_t ccSpaceToDepthForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                 const int32_t blockSize, const void *beta, const ccTensorDescriptor_t outputDesc,
                                 void *output);

/**
 * @ingroup dnn
 * @brief full eltwise forward computation
 * @param [in] handle          cce handle
 * @param [in] eltDesc         eltwise descriptor
 * @param [in] mode            mode of eltwise
 * @param [in] alpha           scaling factors
 * @param [in] broadcast(Reserve) support tensor broadcasting or not
 * @param [in] xDesc[]         array of descriptor for input tensor
 * @param [in] x               array of input data in device memory
 * @param [in] inputNum        the number of input tensors
 * @param [in] beta            scaling factors
 * @param [in] yDesc           descriptor of output tensor
 * @param [in|out] y           output data in device memory
 * @return ccStatus_t
 */

ccStatus_t ccEltwiseForwardEx(ccHandle_t handle, ccEltwiseDescriptor_t eltDesc, ccEltwiseMode_t mode, int32_t inputNum,
                              const void *alpha, bool broadcast, const ccTensorDescriptor_t xDesc[], const void *x[],
                              const void *beta, const ccTensorDescriptor_t yDesc, void *y);
/**
 * @ingroup dnn
 * @brief create descriptor of eltwise operator
 * @param [in|out] eltwiseDesc   point to descriptor of eltwise operator
 * @return ccStatus_t
 */
ccStatus_t ccCreateEltwiseDescriptor(ccEltwiseDescriptor_t *eltDesc);

/**
 * @ingroup dnn
 * @brief destroy eltwise descriptor
 * @param [in] descriptor of eltwise operator
 * @return ccStatus_t
 */
ccStatus_t ccDestroyEltwiseDescriptor(ccEltwiseDescriptor_t *eltDesc);

/**
 * @ingroup dnn
 * @brief append operation after eltwise
 * @param [in|out] eltDesc   descriptor of eltwise operator
 * @param [in] opType  operation type for append at eltwise operation
 * @param [in] opDesc  operation descritpor for the opType
 * @return ccStatus_t
 */
ccStatus_t ccEltwiseAppendOp(ccEltwiseDescriptor_t eltDesc, ccOpType_t opType, const void *opDesc);

/**
 * @ingroup dnn
 * @brief set eltwise desciptor's quantize  parameters
 * @param [in] eltDesc        eltwise descriptor
 * @param [in] quantizeInfo    descriptor of quantize parameters
 * @return ccStatus_t
 */
ccStatus_t ccSetEltwiseQuantizeInfo(ccEltwiseDescriptor_t eltDesc, const ccQuantizeDescriptor_t QuantizeInfo);

/**
 * @ingroup dnn
 * @brief get the temp space size of reshape forward computation, maybe no need temp space
 * @param [in] handle   cce handle
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] yDesc   descriptor of output tensor
 * @param [in|out] sizeInBytes   temp space size need for specified algorithm
 * @return ccStatus_t
 */
ccStatus_t ccGetReshapeForwardWorkspaceSize(ccHandle_t handle, const ccTensorDescriptor_t xDesc,
                                            const ccTensorDescriptor_t yDesc, uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief reshape the input tensor
 * @param [in] handle  cce handle
 * @param [in] alpha   scaling factors
 * @param [in] xDesc   input tensor
 * @param [in] x   input data
 * @param [in] workSpace   temp space, maybe NULL if no need temp space
 * @param [in] workSpaceSizeInBytes   sizeof workspace
 * @param [in] beta   scaling factors
 * @param [in] yDesc   output tensor
 * @param [in|out] y   output data
 * @return ccStatus_t
 */
ccStatus_t ccReshapeForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                            void *workSpace, uint32_t workSpaceSizeInBytes, const void *beta,
                            const ccTensorDescriptor_t yDesc, void *y);
/**
 * @ingroup dnn
 * @brief reshape the input tensor for data in ND format
 * @param [in] handle  cce handle
 * @param [in] alpha   scaling factors
 * @param [in] xDesc   input tensor
 * @param [in] x   input data
 * @param [in] workSpace   temp space, maybe NULL if no need temp space
 * @param [in] workSpaceSizeInBytes   sizeof workspace
 * @param [in] beta   scaling factors
 * @param [in] yDesc   output tensor
 * @param [in|out] y   output data
 * @return ccStatus_t
 */
ccStatus_t ccNdReshapeForward(ccHandle_t handle, const void *alpha, ccTensorFormat_t rawFormat,
                              const ccTensorDescriptor_t xDesc, const void *x, void *workSpace,
                              uint32_t workSpaceSizeInBytes, const void *beta, const ccTensorDescriptor_t yDesc,
                              void *y);

/**
 * @ingroup dnn
 * @brief Four2Five forward computation
 * @param [in] handle          cce handle
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] yDesc           descriptor of output tensor
 * @param [in | out] y         output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccFour2FiveForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                              const void *beta, const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief Five2Four forward computation
 * @param [in] handle          cce handle
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] yDesc           descriptor of output tensor
 * @param [in | out] y         output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccFive2FourForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                              const void *beta, const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief get the temp space size of add forward computation
 * @param [in] handle             cce handle
 * @param [in] xDesc              descriptor of the first input tensor
 * @param [in] wDesc              descriptor of the second input tensor
 * @param [in] yDesc              descriptor of output tensor
 * @param [in|out] sizeInBytes    temp space size need for specified algorithm
 * @return ccStatus_t
 */
ccStatus_t ccGetAddForwardWorkspaceSize(ccHandle_t handle, const ccTensorDescriptor_t xDesc,
                                        const ccTensorDescriptor_t wDesc, const ccTensorDescriptor_t yDesc,
                                        uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief Add forward computation
 * @param [in] handle          cce handle
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               one input data in device memory
 * @param [in] wDesc           descriptor of input tensor
 * @param [in] w               the other input data in device memory
 * @param [in] beta            bias factors
 * @param [in] workSpace       the address apply in HBM
 * @param [in] workSpaceSizeInBytes   the size apply in HBM
 * @param [in] yDesc           descriptor of output tensor
 * @param [in|out] y           output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccAddForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                        const ccTensorDescriptor_t wDesc, const void *w, const void *beta, void *workSpace,
                        uint32_t workSpaceSizeInBytes, const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief Stack forward computation
 * @param [in] handle          cce handle
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x[]             x array is host mem array, the element is device address of input data
 * @param [in] num             number of input tensor
 * @param [in] axis            along which axis to stack the input tensor
 * @param [in] beta            bias factors
 * @param [in] yDesc           descriptor of output tensor
 * @param [in|out] y           output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccStackForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x[],
                          uint32_t num, int32_t axis, const void *beta, const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief get the output dimension info of stack
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] num              number of input tensor
 * @param [in] axis             along which axis to stack the input tensor
 * @param [in|out] n            point to batch size
 * @param [in|out] c            point to channels
 * @param [in|out] h            point to height
 * @param [in|out] w            point to width
 * @param [in|out] realDimCnt   point to real dimCnt after stack
 * @return ccStatus_t
 */
ccStatus_t ccGetStackOutputDim(const ccTensorDescriptor_t xDesc, uint32_t num, int32_t axis, int32_t *n, int32_t *c,
                               int32_t *h, int32_t *w, int32_t *realDimCnt);

/**
 * @ingroup dnn
 * @brief get the output dimension info of stack
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] num              number of input tensor
 * @param [in] axis             along which axis to stack the input tensor
 * @param [in|out] dimCnt       dimcnt
 * @param [in|out] dim          save dim value
 * @param [in| dimlen           length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetStackOutputDim(const ccTensorDescriptor_t xDesc, uint32_t num, int32_t axis, int32_t *dimCnt,
                               int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief return need grid generator or not
 * @param [in] inputH, inputW, outputH, outputW, alignCorner(interp=true,resizeBilinear depends para align corner)
 * @param [out] bool needGridFlag, true mean need, false mean not need
 * @return ccStatus_t
 */
ccStatus_t ccIsGridGenetatorNeed(int32_t inputH, int32_t inputW, int32_t outputH, int32_t outputW, bool alignCorner,
                                 bool &needGridFlag);

/**
 * @ingroup dnn
 * @brief get the temp space size of Deconvolution forward computation, maybe no need temp space
 * @param [in] handle   cce handle
 * @param [in] deconvDesc   descriptor of Deconvolution operator
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] wDesc   descriptor of filter
 * @param [in] yDesc   descriptor of output tensor
 * @param [in] algo   algorithm of Deconvolution forward
 * @param [in|out] sizeInBytes   temp space size need for specified algorithm
 * @return ccStatus_t
 */
ccStatus_t ccGetDeconvolutionForwardWorkspaceSize(ccHandle_t handle, const ccConvolutionDescriptor_t deconvDesc,
                                                  const ccTensorDescriptor_t xDesc, const ccFilterDescriptor_t wDesc,
                                                  const ccTensorDescriptor_t yDesc, ccConvolutionFwdAlgo_t algo,
                                                  uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief Deconvolution forward computation
 * @param [in] handle   cce handle
 * @param [in] deconvDesc   descriptor of deconvolution operator
 * @param [in] alpha   scaling factors
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] x   input data in device memory
 * @param [in] wDesc   descriptor of filter
 * @param [in] w   filter data in device memory
 * @param [in] biasDesc   descriptor of bias
 * @param [in] bias   bias data in device memory
 * @param [in] algo   algorithm of deconvolution forward
 * @param [in] workSpace   temp space, maybe NULL if no need temp space
 * @param [in] workSpaceSizeInBytes   sizeof workspace
 * @param [in] beta   scaling factors
 * @param [in] yDesc   descriptor of output tensor
 * @param [in|out] y   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccDeconvolutionForward(ccHandle_t handle, const ccConvolutionDescriptor_t deconvDesc, const void *alpha,
                                  const ccTensorDescriptor_t xDesc, const void *x, const ccFilterDescriptor_t wDesc,
                                  const void *w, const ccTensorDescriptor_t biasDesc, const void *bias,
                                  ccConvolutionFwdAlgo_t algo, void *workSpace, uint32_t workSpaceSizeInBytes,
                                  const void *beta, const ccTensorDescriptor_t yDesc, void *y);

#define MODE_C_N (0)
#define MODE_N_C (1)

/**
 * [ccArgMaxForward]
 * @param [in] handle        [handle]
 * @param [in] alpha         [reserved parameters]
 * @param [in] xDesc         [x tensor descriptor]
 * @param [in] x             [innput tensor]
 * @param [in] outMaxVaule   [Whether to return the maximum value, true: return max value; false: return max value index
 * ]
 * @param [in] topK          [The number that returns the maximum index or maximum value]
 * @param [in] axis          [Describes which axis of the input Tensor to reduce across]
 * @param [in] beta          [reserved parameters]
 * @param [in] yDesc         [y tensor descriptor]
 * @param [in] y             [The max value index or max value tensor]
 */
ccStatus_t ccArgMaxForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                           bool outMaxVal, uint32_t topK, int32_t axis, const void *beta,
                           const ccTensorDescriptor_t yDesc, void *y);
/**
 * [ccGetArgMaxOutputDim]
 * @param [in] xDesc         [x tensor descriptor]
 * @param [in] outMaxVaule   [Whether to return the maximum value, true: return max value; false: return max value index
 * ]
 * @param [in] topK          [The number that returns the maximum index or maximum value]
 * @param [in] axis          [Describes which axis of the input Tensor to reduce across]
 * @param [in|out] dimCnt    [point to the output dimCnt]
 * @param [in|out] dim       [arrays to save dims]
 * @param [in| dimlen        length of dim
 */
ccStatus_t ccGetArgMaxOutputDim(const ccTensorDescriptor_t xDesc, bool outMaxVal, uint32_t topK, int32_t axis,
                                int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * [ccGetArgMaxOutputDim]
 * @param [in] xDesc         [x tensor descriptor]
 * @param [in] outMaxVaule   [Whether to return the maximum value, true: return max value; false: return max value index
 * ]
 * @param [in] topK          [The number that returns the maximum index or maximum value]
 * @param [in] axis          [Describes which axis of the input Tensor to reduce across]
 * @param [in] n             [Batch number of the output tensor]
 * @param [in] c             [Channel of the output tensor]
 * @param [in] h             [Height number of the output tensor]
 * @param [in] w             [Weight number of the output tensor]
 */
ccStatus_t ccGetArgMaxOutputDim(const ccTensorDescriptor_t xDesc, bool outMaxVal, uint32_t topK, int32_t axis,
                                int32_t *n, int32_t *c, int32_t *h, int32_t *w);

/**
 * @ingroup dnn
 * @brief Yolo2ReorgForward computation
 * @param [in] handle   CCE handle
 * @param [in] stride  scale parameter
 * @param [in] reverse  reverse parameter
 * @param [in] alpha  alpha factor
 * @param [in] beta  beta factor
 * @param [in] xDesc x-tensor descriptor
 * @param [in] x  x-tensor in device memory
 * @param [out] workSpaceSizeInBytes  temporary work sapce size
 * @param [out] workSpace  temporary work sapce in device memory
 * @param [in] yDesc y-tensor descriptor
 * @param [out] y  y-tensor in device memory
 * @return ccStatus_t
 */
ccStatus_t ccYolo2ReorgForward(ccHandle_t handle, int32_t stride, bool reverse, const void *alpha,
                               const ccTensorDescriptor_t xDesc, const void *x, uint32_t workSpaceSizeInBytes,
                               void *workSpace, const void *beta, const ccTensorDescriptor_t yDesc, void *y);
/**
 * @param [in] stride  scale parameter
 * @param [in] reverse  reverse parameter
 * @param [in] xDesc x-tensor descriptor
 * @param [in|out] n          point to batch size
 * @param [in|out] c          point to channels
 * @param [in|out] h          point to height of feature map
 * @param [in|out] w          point to width of feature map
 */
ccStatus_t ccGetReorgOutPutDim(int32_t stride, bool reverse, const ccTensorDescriptor_t xDesc, int32_t *n, int32_t *c,
                               int32_t *h, int32_t *w);

/**
 * @param [in] stride  scale parameter
 * @param [in] reverse  reverse parameter
 * @param [in] xDesc x-tensor descriptor
 * @param [out] dimCnt               output tensor dim cnt
 * @param [out] dim                  output tensor dim
 * @param [in| dimlen           length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetReorgOutPutDim(int32_t stride, bool reverse, const ccTensorDescriptor_t xDesc, int32_t *dimCnt,
                               int32_t dim[], int32_t dimLen);

/**
 * @param [in] xDesc x-tensor descriptor
 * @param [out] temporary work sapce size
 */
ccStatus_t ccGetYolo2ReorgForwardWorkspaceSize(const ccTensorDescriptor_t xDesc, uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief full shuffle       channel forward computation
 * @param [in] handle        cce handle
 * @param [in] groupNum      number of groups in a channal
 * @param [in] subgroupNum   number of sub-groups in a group
 * @param [in] alpha         scaling factors
 * @param [in] xDesc         descriptor of input tensor
 * @param [in] x             input data in device memory
 * @param [in] beta          scaling factors
 * @param [in] yDesc         descriptor of output tensor
 * @param [in|out] y         output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccShuffleChannelForward(ccHandle_t handle, int32_t groupNum, int32_t subgroupNum, const void *alpha,
                                   const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                   const ccTensorDescriptor_t yDesc, void *y);
/**
 * @ingroup dnn
 * @brief get the temp space size of permute forward computation, maybe no need temp space
 * @param [in] handle   cce handle
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] yDesc   descriptor of output tensor
 * @param [in|out] sizeInBytes   temp space size need for specified algorithm
 * @return ccStatus_t
 */
ccStatus_t ccGetPermuteForwardWorkspaceSize(ccHandle_t handle, const ccTensorDescriptor_t xDesc,
                                            const ccTensorDescriptor_t yDesc, uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief get the output dim of permute forward computation
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] dimIndex    dim Index
 * @param [in|out] dimCnt  dim count
 * @param [in|out] dim     dim value
 * @param [in| dimlen      length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetPermuteOutputDim(const ccTensorDescriptor_t xDesc, const int32_t dimIndex[], const int32_t dimIndexLen,
                                 int32_t *dimCnt, int32_t *dim, int32_t dimLen);
/**
 * @ingroup dnn
 * @brief full permute     forward computation
 * @param [in] handle      cce handle
 * @param [in] dimIndex    dim Index,only support [0,1,2,3]
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] x           input data in device memory
 * @param [in] workSpace   temp space, maybe NULL if no need temp space
 * @param [in] workSpaceSizeInBytes   sizeof workspace
 * @param [in] beta        scaling factors
 * @param [in] yDesc       descriptor of output tensor
 * @param [in|out] y       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccPermuteForward(ccHandle_t handle, const int32_t dimIndex[], const void *alpha,
                            const ccTensorDescriptor_t xDesc, const void *x, void *workspace,
                            uint32_t workSpaceSizeInBytes, const void *beta, const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief full split      forward computation
 * @param [in] handle     cce handle
 * @param [in] alpha      scaling factors
 * @param [in] xDesc      descriptor of input tensor
 * @param [in] x          input data in device memory
 * @param [in] axis       the dimension along which to split. Must be in the range [-xDesc->dimCnt, xDesc->dimCnt)
 * @param [in] num        the number of outputs
 * @param [in] beta       scaling factors
 * @param [in] yDescArr      descriptors of output tensors
 * @param [in|out] yArr      output data array in device memory
 * @return ccStatus_t
 */
ccStatus_t ccSplitForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          int32_t axis, uint32_t num, const void *beta, const ccTensorDescriptor_t yDescArr[],
                          void *yArr[]);

/**
 * @ingroup dnn
 * @brief get the output dimensions info of split
 * @param [in] xDesc      descriptor of input tensor
 * @param [in] axis       the dimension along which to split. Must be in the range [-xDesc->dimCnt, xDesc->dimCnt)
 * @param [in] num        the number of outputs
 * @param [in] sizes      Optional, used to specify the sizes of each output tensor along split dim. The tensor x would
 * be split evenly along split dim if sizes is NULL
 * @param [in|out] nArr   point to the first element of batch sizes
 * @param [in|out] cArr   point to the first element of channels
 * @param [in|out] hArr   point to the first element of heights of feature map
 * @param [in|out] wArr   point to the first element of widths of feature map
 * @return ccStatus_t
 */
ccStatus_t ccGetSplitForwardOutputDim(const ccTensorDescriptor_t xDesc, int32_t axis, uint32_t num,
                                      const uint32_t sizes[], uint32_t nArr[], uint32_t cArr[], uint32_t hArr[],
                                      uint32_t wArr[]);

/**
    * @ingroup dnn
    * @brief Get split output shape(s).
    * @param [in] xDesc         input tensor, support ND and NC1HWC0
    * @param [in] axis          split axis, negtive axis will increased by dimCnt once time.
    * @param [in] num           splited nums.
    * @param [in] sizes         splited dim size on axis. if NULL was set, The input will be divided into num equally.
    * @param [output] dimCnt    splited dimCnt array. One to one correspondence with the splited output.
    * @param [output] dim       array of splited dim array. One to one correspondence with the splited output.
    * @param [in| dimlen        length of dim(Pass in the length of the entire space pointed to by dim,
                                              not just the length of the dim array, because dim is a level 2 array
                                              dimlen = lengthof dim[][], not just lengthof dim[])
    * @return ccStatus_t
    */
ccStatus_t ccGetSplitForwardOutputDim(const ccTensorDescriptor_t xDesc, int32_t axis, uint32_t num,
                                      const uint32_t sizes[], int32_t *dimCnt, int32_t *dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief create weight compress info
 * @param [in|out] compressInfo   point to CompressInfo
 * @return ccStatus_t
 */
ccStatus_t ccCreateWeightCompressInfo(ccWeightCompressInfo_t **compressInfo);

/**
 * @ingroup dnn
 * @brief destory weight compress info
 * @param [in] *compressInfo   point to CompressInfo
 * @return ccStatus_t
 */
ccStatus_t ccDestroyWeightCompressInfo(ccWeightCompressInfo_t **compressInfo);

/**
 * @ingroup dnn
 * @brief create compress table
 * @param [in|out] compressTab   point to weight compress table
 * @return ccStatus_t
 */
ccStatus_t ccCreateWeightCompressTab(ccWeightCompressTab_t **compressTab);

/**
 * @ingroup dnn
 * @brief destory compress table
 * @param [in] compressTab   point to weight compress table
 * @return ccStatus_t
 */
ccStatus_t ccDestroyWeightCompressTab(ccWeightCompressTab_t **compressTab);

/**
 * @ingroup dnn
 * @brief get fc compress info
 * @param [in] xDesc               descriptor of input tensor
 * @param [in] wDesc               descriptor of weight tensor
 * @param [in] biasDesc            descriptor of bias tensor
 * @param [in] dataTypeTransmode   mode of data type transform
 * @param [in] weightCompressInfo  compress info, compute based on tiling method
 * @param [in|out] outputSize      output data size in byte
 * @param [in|out] infoTabSize     compress info table
 * @return ccStatus_t
 */
ccStatus_t ccGetCompressedFcWeightInfo(const ccTensorDescriptor_t xDesc, const ccFilterDescriptor_t wDesc,
                                       const ccTensorDescriptor_t biasDesc, ccDataTypeTransMode_t dataTypeTransmode,
                                       ccWeightCompressInfo_t *weightCompressInfo, uint32_t *outputSize,
                                       uint32_t *infoTabSize);
/**
 * @ingroup dnn
 * @brief compress fc
 * @param [in] wDesc               descriptor of weight tensor
 * @param [in] w                   filter data in device memory
 * @param [in] weightCompressInfo  compress info, compute based on tiling method
 * @param [in] dataTypeTransmode   mode of data type transform
 * @param [in|out] y               output data in device memory
 * @param [in] ySize               transformed data size in byte
 * @param [in|out] yCompressedSize compressed output data size in byte
 * @param [in|out] infoTab         compressed info table
 * @param [in] infoTabSize         compressed info table size in byte
 * @return ccStatus_t
 */
ccStatus_t ccCompressWeight(const ccFilterDescriptor_t wDesc, const void *w,
                            const ccWeightCompressInfo_t *weightCompressInfo, ccDataTypeTransMode_t dataTypeTransmode,
                            ccFilterDescriptor_t yDesc, void *y, uint32_t ySize, uint32_t *yCompressedSize,
                            void *infoTab, uint32_t infoTabSize);

/**
 * @ingroup dnn
 * @brief restore compressed fc data
 * @param [in] x               input data in device memory
 * @param [in] xSizeInBytes    input compressed weight data size in byte
 * @param [in|out] y           output data in device memory
 * @param [in] ySizeInBytes    output data size in byte
 * @return ccStatus_t
 */
ccStatus_t ccRestoreCompressedWeight(const void *x, uint32_t xSizeInBytes, void *y, uint32_t ySizeInBytes,
                                     rtMemcpyKind_t kind);

/**
 * @ingroup dnn
 * @brief create quantize parameters struct
 * @param [in|out] quantizeInfo    descriptor of quantize parameters
 * @return ccStatus_t
 */
ccStatus_t ccCreateQuantizeInfoTab(ccQuantizeDescriptor_t *quantizeInfo);

/**
 * @ingroup dnn
 * @brief destroy quantize parameters struct
 * @param [in] quantizeInfo    descriptor of quantize parameters
 * @return ccStatus_t
 */
ccStatus_t ccDestoryQuantizeInfoTab(ccQuantizeDescriptor_t *quantizeInfo);

/**
 * @ingroup dnn
 * @brief set quantize parameters
 * @param [in] quantizeInfo    descriptor of quantize parameters
 * @param [in] scaleValMode    enmu type for quantize scale value type (normal or sqrt)
 * @param [in] scale           quantize scale value
 * @param [in] offset          quantize offset(when quantize algorithm is half offset or full offset,this should be
 * configed)
 * @param [in] offsetPad       padding value for load3d (only for half offset or full offset)
 * @return ccStatus_t
 */
ccStatus_t ccSetQuantizeFactors(ccQuantizeDescriptor_t quantizeInfo, ccScaleValueMode_t scaleValMode,
                                const uint16_t *scale, const uint16_t *offset, const uint8_t *offsetPad);

/**
 * @ingroup dnn
 * @brief set Requantize parameters
 * @param [in] quantizeInfo    descriptor of quantize parameters
 * @param [in] scaleValMode    enmu type for requantize scale value type (normal or sqrt)
 * @param [in] scale           quantize scale value
 * @param [in] offset          quantize offset(when quantize algorithm is half offset or full offset,this should be
 * configed)
 * @param [in] offsetw         offset for filter (only config for full offset quantize)
 * @return ccStatus_t
 */
ccStatus_t ccSetReQuantizeFactors(ccQuantizeDescriptor_t quantizeInfo, ccScaleValueMode_t scaleValMode,
                                  const uint16_t *scaleRq, const uint16_t *nextLayerOffset, const int32_t *offsetw);

/**
 * @ingroup dnn
 * @brief set Dequantize parameters
 * @param [in] quantizeInfo    descriptor of quantize parameters
 * @param [in] scaleValMode    enmu type for dequantize scale value type (normal or sqrt)
 * @param [in] scaleDq           quantize scale value
 * @param [in] offsetw         offset for filter (only config for full offset quantize)
 * @return ccStatus_t
 */
ccStatus_t ccSetDeQuantizeFactors(ccQuantizeDescriptor_t quantizeInfo, ccScaleValueMode_t scaleValMode,
                                  const uint16_t *scaleDq, const int32_t *offsetw);

/**
 * @ingroup dnn
 * @brief set convolution desciptor's quantize  parameters
 * @param [in] convDesc        convolution descriptor
 * @param [in] quantizeInfo    descriptor of quantize parameters
 * @return ccStatus_t
 */
ccStatus_t ccSetConvolutionQuantizeInfo(ccConvolutionDescriptor_t convDesc, const ccQuantizeDescriptor_t QuantizeInfo);

/**
 * @ingroup dnn
 * @brief set convolution desciptor's all offset quantize  parameters
 * @param [in] convDesc        convolution descriptor
 * @param [in] offsetw         descriptor of quantize parameters
 * @param [in] scaleReq        descriptor of quantize parameters
 * @param [in] offset_d_next   descriptor of quantize parameters
 * @return ccStatus_t
 */
ccStatus_t ccSetAllOffsetQuantizeFactors(ccQuantizeDescriptor_t quantizeInfo, const uint8_t *offsetW,
                                         const uint8_t *offsetD, const uint16_t *scaleReq, const uint16_t *offsetDNext);

/**
 * @ingroup dnn
 * @brief set full connection desciptor's quantize  parameters
 * @param [in] fcDesc          full connection descriptor
 * @param [in] quantizeInfo    descriptor of quantize parameters
 * @return ccStatus_t
 */
ccStatus_t ccSetFullConnectionQuantizeInfo(ccFullConnectionDescriptor_t fcDesc,
                                           const ccQuantizeDescriptor_t QuantizeInfo);

/**
 * @ingroup dnn
 * @brief set pooling desciptor's quantize  parameters
 * @param [in] poolingDesc     pooling descriptor
 * @param [in] quantizeInfo    descriptor of quantize parameters
 * @return ccStatus_t
 */
ccStatus_t ccSetPoolingQuantizeInfo(ccPoolingDescriptor_t poolingDesc, const ccQuantizeDescriptor_t QuantizeInfo);

/**
 * @ingroup dnn
 * @brief  set full connection  desciptor's info table
 * @param [in] fcDesc          full connection descriptor
 * @param [in] infoTabSize     table size
 * @param [in] infoTab         pointer to info table
 * @return ccStatus_t
 */
ccStatus_t ccSetFullConnectionDescriptor(ccFullConnectionDescriptor_t fcDesc, uint32_t infoTabSize, const void *infoTab,
                                         ccFullConnectFwdAlgo_t algo = CC_FULLCONNECT_FWD_ALGO_HALF);

/**
 * @ingroup dnn
 * @brief  set full connection  desciptor's relu flag
 * @param [in] fcDesc          full connection descriptor
 * @param [in] opType  operation type for append at convolution operation
 * @param [in] opDesc  operation descritpor for the opType
 * @return ccStatus_t
 */
ccStatus_t ccFullConnectionAppendOp(ccFullConnectionDescriptor_t fcDesc, tagCcOpType opType, const void *opDesc);

/**
 * @ingroup dnn
 * @brief check aipp basic info
 * @param [in] inputFormat     format of input image
 * @param [in] loadStartPosH   vertical start position in source image
 * @param [in] loadStartPosW   horizontal start position in source image
 * @param [in] srcImageSizeH   vertical size of source image
 * @param [in] srcImageSizeW   horizontal size of source image
 * @param [in] cpaddingValue   C direction padding value
 * @param [in] cscSwitch       csc enable or not
 * @param [in] rbuvSwapSwitch  swap R/U and B/V position of the image
 * @param [in] axSwapSwitch    swap RGBA->ARGB, YUVA->AYUV
 * @param [in] singleLineMode  when set this bit to 1, only read 1 line. Under this case, vertical size configuration is
 * not useful.
 * @return ccStatus_t
 */
ccStatus_t ccCheckConvolutionAippCommInfo(ccAippInputFormat_t inputFormat, int32_t loadStartPosW, int32_t loadStartPosH,
                                          int32_t srcImageSizeW, int32_t srcImageSizeH, float cpaddingValue,
                                          bool cscSwitch, bool rbuvSwapSwitch, bool axSwapSwitch, bool singleLineMode);

/**
 * @ingroup dnn
 * @brief check aipp dtc info
 * @param [in] dtcPixelMeanChnx      Mean value for YUV or RGB data channel x
 * @param [in] dtcPixelMinChnx       Min value for YUV or RGB data channel x
 * @param [in] dtcPixelVarReciChnx   Reciprocal of variance or (max-min) for YUV or RGB data channel x
 * @return ccStatus_t
 */
ccStatus_t ccCheckConvolutionAippDtcInfo(int32_t dtcPixelMeanChn0, int32_t dtcPixelMeanChn1, int32_t dtcPixelMeanChn2,
                                         float dtcPixelMinChn0, float dtcPixelMinChn1, float dtcPixelMinChn2,
                                         float dtcPixelVarReciChn0, float dtcPixelVarReciChn1,
                                         float dtcPixelVarReciChn2);

/**
 * @ingroup dnn
 * @brief check aipp pad info
 * @param [in] paddingMode              padding mode
 * @param [in] leftPaddingSize          left hblank/padding size
 * @param [in] rightPaddingSize         right hblank/padding size
 * @param [in] topPaddingSize           top padding size
 * @param [in] bottomPaddingSize        bottom padding size
 * @return ccStatus_t
 */
ccStatus_t ccCheckConvolutionAippPadInfo(ccAippPaddingMode_t paddingMode, int32_t leftPaddingSize,
                                         int32_t rightPaddingSize, int32_t topPaddingSize, int32_t bottomPaddingSize);

/**
 * @ingroup dnn
 * @brief check aipp csc info
 * @param [in] cscMatrixRmCn           3x3 CSC matrix for YUV to RGB or RGB to YUV, element of row m and column n
 * @param [in] cscOutputBiasm          output Bias for RGB to YUV, element of row m
 * @param [in] cscInputBiasm           input Bias for YUV to RGB, element of row m
 * @return ccStatus_t
 */
ccStatus_t ccCheckConvolutionAippCscInfo(int32_t cscMatrixR0C0, int32_t cscMatrixR0C1, int32_t cscMatrixR0C2,
                                         int32_t cscMatrixR1C0, int32_t cscMatrixR1C1, int32_t cscMatrixR1C2,
                                         int32_t cscMatrixR2C0, int32_t cscMatrixR2C1, int32_t cscMatrixR2C2,
                                         int32_t cscOutputBias0, int32_t cscOutputBias1, int32_t cscOutputBias2,
                                         int32_t cscInputBias0, int32_t cscInputBias1, int32_t cscInputBias2);

/**
 * @ingroup dnn
 * @brief check aipp scf info
 * @param [in] scfSwitch               scaling enable or not
 * @param [in] scfInputW               input width of scaling
 * @param [in] scfInputH               input height of scaling
 * @param [in] scfOutputW              output width of scaling
 * @param [in] scfOutputH              output height of scaling
 * @return ccStatus_t
 */
ccStatus_t ccCheckConvolutionAippScfInfo(bool scfSwitch, int32_t scfInputW, int32_t scfInputH, int32_t scfOutputW,
                                         int32_t scfOutputH);

/**
 * @ingroup dnn
 * @brief check aipp param
 * @param [in] convDesc                descriptor of conv operator
 * @param [in] xDesc                   input tensor info
 * @param [in] yDesc                   output tensor info
 * @return ccStatus_t
 */
ccStatus_t ccCheckConvFwdAippParam(const ccConvolutionDescriptor_t convDesc, const ccTensorDescriptor_t xDesc,
                                   const ccTensorDescriptor_t yDesc);

/**
 * @ingroup dnn
 * @brief init aipp basic info
 * @param [in|out] convDesc   descriptor of conv operator
 * @param [in] inputFormat     format of input image
 * @param [in] loadStartPosH   vertical start position in source image
 * @param [in] loadStartPosW   horizontal start position in source image
 * @param [in] srcImageSizeH   vertical size of source image
 * @param [in] srcImageSizeW   horizontal size of source image
 * @param [in] cpaddingValue   C direction padding value
 * @param [in] cscSwitch       csc enable or not
 * @param [in] rbuvSwapSwitch  swap R/U and B/V position of the image
 * @param [in] axSwapSwitch    swap RGBA->ARGB, YUVA->AYUV
 * @param [in] singleLineMode  when set this bit to 1, only read 1 line. Under this case, vertical size configuration is
 * not useful.
 * @return ccStatus_t
 */
ccStatus_t ccSetConvolutionAippCommInfo(ccConvolutionDescriptor_t convDesc, ccAippInputFormat_t inputFormat,
                                        int32_t loadStartPosW, int32_t loadStartPosH, int32_t srcImageSizeW,
                                        int32_t srcImageSizeH, float cpaddingValue, bool cscSwitch, bool rbuvSwapSwitch,
                                        bool axSwapSwitch, bool singleLineMode);
/**
 * @ingroup dnn
 * @brief init aipp dtc info
 * @param [in|out] convDesc   descriptor of conv operator
 * @param [in] dtcPixelMeanChnx      Mean value for YUV or RGB data channel x
 * @param [in] dtcPixelMinChnx       Min value for YUV or RGB data channel x
 * @param [in] dtcPixelVarReciChnx   Reciprocal of variance or (max-min) for YUV or RGB data channel x
 * @return ccStatus_t
 */
ccStatus_t ccSetConvolutionAippDtcInfo(ccConvolutionDescriptor_t convDesc, int32_t dtcPixelMeanChn0,
                                       int32_t dtcPixelMeanChn1, int32_t dtcPixelMeanChn2, float dtcPixelMinChn0,
                                       float dtcPixelMinChn1, float dtcPixelMinChn2, float dtcPixelVarReciChn0,
                                       float dtcPixelVarReciChn1, float dtcPixelVarReciChn2);
/**
 * @ingroup dnn
 * @brief init aipp pad info
 * @param [in|out] convDesc   descriptor of conv operator
 * @param [in] paddingMode              padding mode
 * @param [in] leftPaddingSize          left hblank/padding size
 * @param [in] rightPaddingSize         right hblank/padding size
 * @param [in] topPaddingSize           top padding size
 * @param [in] bottomPaddingSize        bottom padding size
 * @return ccStatus_t
 */
ccStatus_t ccSetConvolutionAippPadInfo(ccConvolutionDescriptor_t convDesc, ccAippPaddingMode_t paddingMode,
                                       int32_t leftPaddingSize, int32_t rightPaddingSize, int32_t topPaddingSize,
                                       int32_t bottomPaddingSize);

/**
 * @ingroup dnn
 * @brief init aipp csc info
 * @param [in|out] convDesc   descriptor of conv operator
 * @param [in] cscMatrixRmCn           3x3 CSC matrix for YUV to RGB or RGB to YUV, element of row m and column n
 * @param [in] cscOutputBiasm          output Bias for RGB to YUV, element of row m
 * @param [in] cscInputBiasm           input Bias for YUV to RGB, element of row m
 * @return ccStatus_t
 */
ccStatus_t ccSetConvolutionAippCscInfo(ccConvolutionDescriptor_t convDesc, int32_t cscMatrixR0C0, int32_t cscMatrixR0C1,
                                       int32_t cscMatrixR0C2, int32_t cscMatrixR1C0, int32_t cscMatrixR1C1,
                                       int32_t cscMatrixR1C2, int32_t cscMatrixR2C0, int32_t cscMatrixR2C1,
                                       int32_t cscMatrixR2C2, int32_t cscOutputBias0, int32_t cscOutputBias1,
                                       int32_t cscOutputBias2, int32_t cscInputBias0, int32_t cscInputBias1,
                                       int32_t cscInputBias2);

/**
 * @ingroup dnn
 * @brief init aipp scf info
 * @param [in|out] convDesc   descriptor of conv operator
 * @param [in] scfSwitch               scaling enable or not
 * @param [in] scfInputW               input width of scaling
 * @param [in] scfInputH               input height of scaling
 * @param [in] scfOutputW              output width of scaling
 * @param [in] scfOutputH              output height of scaling
 * @return ccStatus_t
 */
ccStatus_t ccSetConvolutionAippScfInfo(ccConvolutionDescriptor_t convDesc, bool scfSwitch, int32_t scfInputW,
                                       int32_t scfInputH, int32_t scfOutputW, int32_t scfOutputH);

/**
 * @ingroup dnn
 * @brief set dynamic aipp parameter address and enflag info
 * @param [in|out] convDesc   descriptor of conv operator
 * @param [in] dyncParaAddr            aipp parameter address
 * @param [in] dyncAippFlag            flag to show whether to use dynamic aipp
 * @return ccStatus_t
 */
ccStatus_t ccSetConvolutionAippDyncParaAddr(ccConvolutionDescriptor_t convDesc, const void *dyncParaAddr,
                                            bool dyncAippFlag, bool rotationFlag = false);

/**
 * @ingroup dnn
 * @brief check dynamic aipp parameter
 * @param [in] dyncParaAddr            aipp parameter address
 * @param [in] dataLength              parameter lenght
 * @param [in] convolutionDimW            convDimW
 * @param [in] convolutionDimH            convDimH
 * @return ccStatus_t
 */
ccStatus_t ccCheckDynamicAippParam(const void *dynamicParamAddr, uint32_t dataLength, int64_t convolutionDimW,
                                   int64_t convolutionDimH);

/*** @ingroup dnn
 * @brief trans mean and var
 * @param [in|out] mean' = bnScale/sqrt(var)
 * @param [in|out] var' = -bnScale * mean / sqrt(var) + bnBias
 * @return ccStatus_t
 */

ccStatus_t ccTransBatchnormMeanAndVar(void *mean, void *var, const ccTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                      const void *alpha, const void *beta, void *bnScale, void *bnBias, double epsilon);

/**
 * @ingroup dnn
 * @brief init deconvolution adj or targetShape info.
 * @param [in] convDesc  conv descriptor.
 * @param [in] adjH, adjust H output.
 * @param [in] adjW, adjust W output.
 * @param [in] targetShape, values of output shape, if this pointer was set, ignore adj.
 * @return ccStatus_t
 */
ccStatus_t ccSetDeconvolutionOutShapeInfo(ccConvolutionDescriptor_t convDesc, uint32_t adjSize, const uint32_t *adj,
                                          uint32_t targetShapeSize, const uint32_t *targetShape);

/**
 * @ingroup dnn
 * @brief gather elements according to the indices.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the tensor from which to gather elements.
 * @param [in] x  data point of the tensor from which to gather elements.
 * @param [in] indicesDesc  description of the tensor of indices.
 * @param [in] indices  data point of the tensor of indices.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccGatherNdForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                             const ccTensorDescriptor_t indicesDesc, const void *indices, const void *beta,
                             const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get output shape of gather_nd.
 * @param [in] xDesc  description of the tensor from which to gather elements.
 * @param [in] indicesDesc  description of the tensor of indices.
 * @param [output] n dim-size of n-dim.
 * @param [output] c dim-size of c-dim.
 * @param [output] h dim-size of h-dim.
 * @param [output] w dim-size of w-dim.
 * @param [output] realDimCnt real dim.
 * @return ccStatus_t
 */
ccStatus_t ccGetGatherNdOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t indicesDesc, int32_t *n,
                                  int32_t *c, int32_t *h, int32_t *w, int32_t *realDimCnt);
/**
 * @ingroup dnn
 * @brief get output shape of realdiv.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetGatherNdOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t indicesDesc,
                                  int32_t *dimCnt, int32_t *dim, int32_t dimLen);
/**
 * @ingroup dnn
 * @brief tile tensor by multiples.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the tensor which to be tiled.
 * @param [in] x  data point of the tensor which to be tiled.
 * @param [in] multiples tile coefficient of each dim.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccTileForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                         const ccIntArray_t *multiples, const void *beta, const ccTensorDescriptor_t outputDesc,
                         void *output);

/**
 * @ingroup dnn
 * @brief get output shape of tile.
 * @param [in] xDesc  description of the dividend tensor.
 * @param [in] multiples  multiples of each dim.
 * @param [in|out] dimCnt    [point to the output dimCnt]
 * @param [in|out] dim       [arrays to save dims]
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetTileOutputDim(const ccTensorDescriptor_t xDesc, const ccIntArray_t *multiples, int32_t *dimCnt,
                              int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief get output shape of tile.
 * @param [in] xDesc  description of the dividend tensor.
 * @param [in] multiples  multiples of each dim.
 * @param [output] n dim-size of n-dim.
 * @param [output] c dim-size of c-dim.
 * @param [output] h dim-size of h-dim.
 * @param [output] w dim-size of w-dim.
 * @param [output] realDimCnt real dim.
 * @return ccStatus_t
 */
ccStatus_t ccGetTileOutputDim(const ccTensorDescriptor_t xDesc,
                              // const ccIntArrayDescriptor_t multiples,
                              const ccIntArray_t *multiples, int32_t *n, int32_t *c, int32_t *h, int32_t *w,
                              int32_t *realDimCnt);
/**
 * @ingroup dnn
 * @brief get output shape of realdiv.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetRealdivOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                                 int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief realdiv between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the dividend tensor.
 * @param [in] x  data point of the dividend tensor.
 * @param [in] yDesc  description of the divisor tensor.
 * @param [in] y  data point of the divisor tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccRealdivForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                            const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                            const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get output shape of realdiv.
 * @param [in] xDesc  description of the dividend tensor.
 * @param [in] yDesc  description of the divisor tensor.
 * @param [output] n dim-size of n-dim.
 * @param [output] c dim-size of c-dim.
 * @param [output] h dim-size of h-dim.
 * @param [output] w dim-size of w-dim.
 * @param [output] realDimCnt real dim.
 * @return ccStatus_t
 */
ccStatus_t ccGetRealdivOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *n,
                                 int32_t *c, int32_t *h, int32_t *w, int32_t *realDimCnt);

/**
 * @ingroup dnn
 * @brief realdiv between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccFloordivForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                             const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                             const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get output shape of realdiv.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] realDimCnt real dim.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetFloordivOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                                  int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief realdiv between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccGreaterForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                            const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                            const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get output shape of realdiv.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetGreaterOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                                 int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief realdiv between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccLessForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                         const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                         const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get output shape of realdiv.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetLessOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                              int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief get output shape of LogicalOr.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetLogicalOrOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                                   int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief get output shape of LogicalXor.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in] dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetLogicalXorOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                                    int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief sqrt forward:
 * data type only support bool
 * data format only support ND
 * @param [in] handle cce handle
 * @param [in] alpha common scale factor
 * @param [in] xDesc descriptor of input data
 * @param [in] x input data in device memory
 * @param [in] beta common scale factor
 * @param [in] outputDesc descriptor of output data
 * @param [in|out] output output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccLogicalNotForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                               const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief equal between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */

ccStatus_t ccEqualForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                          const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief dump data during inference, only for eng ver.
 * @param [in] handle        cce handle
 * @return ccStatus_t
 */
ccStatus_t ccDataDumpForward(ccHandle_t handle, const void *buffer, const uint64_t bufLen, const uint32_t taskIndex);

/**
 * @ingroup dnn
 * @brief logicaland between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccLogicalAndForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                               const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                               const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief logical or between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccLogicalOrForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                              const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                              const ccTensorDescriptor_t outputDesc, void *output);
/**
 * @ingroup dnn
 * @brief logical Xor between two tensors(x ^ y = (x | y) & ~(x & y).
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccLogicalXorForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                               const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                               const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get output shape of equal.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetEqualOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                               int32_t *dim, int32_t dimLen);
/**
 * @ingroup dnn
 * @brief get output shape of logicaland.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetLogicalAndOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                                    int32_t *dim, int32_t dimLen);
/**
 * @ingroup dnn
 * @brief realdiv between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccFloormodForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                             const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                             const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get output shape of realdiv.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetFloormodOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                                  int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief compare between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */
ccStatus_t ccCompareForward(ccHandle_t handle, ccCompareType_t compareType, const void *alpha,
                            const ccTensorDescriptor_t xDesc, const void *x, const ccTensorDescriptor_t yDesc,
                            const void *y, const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get output shape of realdiv.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [output] dimCnt dim nums.
 * @param [output] dim dim size.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetCompareOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                                 int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief create descriptor of FillParam
 * @param [in|out] fillParamDesc   point to descriptor of fill param
 * @return ccStatus_t
 */
ccStatus_t ccCreateFillParamDescriptor(ccFillParamDescriptor_t *fillParamDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of FillParam
 * @param [in] *fillParamDesc   point to descriptor of fill param
 * @return ccStatus_t
 */
ccStatus_t ccDestroyFillParamDescriptor(ccFillParamDescriptor_t *fillParamDesc);

/**
 * @ingroup dnn
 * @brief get output shape of broadcat operations.
 * @param [in] inputNum  input number of the operation tensors.
 * @param [in] xDesc[]  description of the input operation tensors list.
 * @param [output] dimCnt dim-size of output tensor.
 * @param [output] dim dim of output tensor.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetMultiNdBroadcastOpOutputDim(const int32_t inputNum, const ccTensorDescriptor_t xDesc[], int32_t *dimCnt,
                                            int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief get output shape of maximultitensor.
 * @param [in] inputNum  the num of input operator tensors.
 * @param [in] xDesc[]  description of the input operator tensors list.
 * @param [output] dimCnt dim count of output tensor.
 * @param [output] dim array of output tensor.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetMaxMultitensorOutputDim(const int32_t inputNum, const ccTensorDescriptor_t xDesc[], int32_t *dimCnt,
                                        int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief get output shape of minmultitensor.
 * @param [in] inputNum  the num of input operator tensors.
 * @param [in] xDesc[]  description of the input operator tensors list.
 * @param [output] dimCnt dim count of output tensor.
 * @param [output] dim array of output tensor.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetMinMultitensorOutputDim(const int32_t inputNum, const ccTensorDescriptor_t xDesc[], int32_t *dimCnt,
                                        int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief MaxMultitensor forward:
 *          data type only support float float16 and int32
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] inputNum     input tensor number
 * @param [in] alpha        common scale factor
 * @param [in] xDesc[]      descriptor of input tensors list
 * @param [in] x[]          input data in device memory list
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccMaxMultitensorForward(const ccHandle_t handle, const int32_t inputNum, const void *alpha,
                                   const ccTensorDescriptor_t xDesc[], const void *x[], const void *beta,
                                   const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief MinMultitensor forward:
 *          data type only support float float16 and int32
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] inputNum     input tensor number
 * @param [in] alpha        common scale factor
 * @param [in] xDesc[]      descriptor of input data list
 * @param [in] x[]          input data in device memory list
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccMinMultitensorForward(const ccHandle_t handle, const int32_t inputNum, const void *alpha,
                                   const ccTensorDescriptor_t xDesc[], const void *x[], const void *beta,
                                   const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief create descriptor of StridedSlice
 * @param [in|out] stridedSliceDesc   point to descriptor of StridedSlice param
 * @return ccStatus_t
 */
ccStatus_t ccCreateStridedSliceDescriptor(ccStridedSliceDescriptor_t *stridedSliceDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of StridedSlice
 * @param [in] *stridedSliceDesc   point to descriptor of StridedSlice param
 * @return ccStatus_t
 */
ccStatus_t ccDestroyStridedSliceDescriptor(ccStridedSliceDescriptor_t *stridedSliceDesc);

/**
 * @ingroup dnn
 * @brief init stridedSlice descriptor_t.
 * @param [out] stridedSliceDesc   struct of stridedslice param
 * @param [in] dimCnt    dimension of the input tensor
 * @param [in] begin     slice begin(include)
 * @param [in] end       slice end index(not include)
 * @param [in] strides   slice stride
 * @return ccStatus_t
 */
ccStatus_t ccSetStridedSliceDescriptor(ccStridedSliceDescriptor_t stridedSliceDesc, int32_t dimCnt, int32_t begin[],
                                       int32_t end[], int32_t strides[]);

/**
 * @ingroup dnn
 * @brief create descriptor of StridedSlice
 * @param [in|out] stridedSliceDesc   point to descriptor of StridedSlice attr
 * @return ccStatus_t
 */
ccStatus_t ccCreateStridedSliceAttrsDescriptor(ccStridedSliceAttrsDescriptor_t *attrDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of StridedSlice
 * @param [in] *stridedSliceDesc   point to descriptor of StridedSlice attr
 * @return ccStatus_t
 */
ccStatus_t ccDestroyStridedSliceAttrsDescriptor(ccStridedSliceAttrsDescriptor_t *attrDesc);

/**
 * @ingroup dnn
 * @brief init stridedSlice mask attrs desescriptor.
 * @param [out] attrDesc   struct of stridedslice mask attrs
 * @param [in] beginMask     begin mask
 * @param [in] endMask       end mask
 * @param [in] ellipsisMask  ellipsis mask
 * @param [in] newAxisMask   new axis mask
 * @param [in] shrinkAxisMask  shrink axis mask
 * @return ccStatus_t
 */
ccStatus_t ccSetStridedSliceAttrsDescriptor(ccStridedSliceAttrsDescriptor_t attrDesc, int32_t beginMask,
                                            int32_t endMask, int32_t ellipsisMask, int32_t newAxisMask,
                                            int32_t shrinkAxisMask);

/**
 * @ingroup dnn
 * @brief Extracts a strided slice of a tensor.
 * @param [in] xDesc   descriptor of input data
 * @param [in] stridedSliceDesc specifies the begin, end, strides of slice
 * @param [in] attrDesc  reserve for optional attributes.
 * @param [out] n       point to n size
 * @param [out] c       point to c size
 * @param [out] h       point to h size
 * @param [out] w       point to w size
 * @return ccStatus_t
 */
ccStatus_t ccGetStridedSliceOutputDim(const ccTensorDescriptor_t xDesc,
                                      const ccStridedSliceDescriptor_t stridedSliceDesc,
                                      const ccStridedSliceAttrsDescriptor_t attrDesc, int32_t *n, int32_t *c,
                                      int32_t *h, int32_t *w, int32_t *realDimCnt);

/**
 * @ingroup dnn
 * @brief Extracts a strided slice of a tensor.
 * @param [in] handle  cce handle
 * @param [in] stridedSliceDesc specifies the  begin, end, strides of slice
 * @param [in] attrDesc  reserve for optional attributes.
 * @param [in] alpha   common scale factor
 * @param [in] xDesc   descriptor of input data
 * @param [in] x   input data in device memory
 * @param [in] beta    common scale factor
 * @param [in] yDesc   descriptor of output data
 * @param [in|out] y   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccStridedSliceForward(ccHandle_t handle, const ccStridedSliceDescriptor_t stridedSliceDesc,
                                 const ccStridedSliceAttrsDescriptor_t attrDesc, const void *alpha,
                                 const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                 const ccTensorDescriptor_t yDesc, void *y);

/**
 * @
 * @brief get out put descrition of slice tensor.
 * @param [in] xDesc         descriptor of input data
 * @param [in] begin         begin position of tensor
 * @param [in] size          size to slice
 * @param [out] n            point to n size
 * @param [out] c            point to c size
 * @param [out] h            point to h size
 * @param [out] w            point to w size
 * @param [out] realDimCnt   realdim count
 * @return ccStatus_t
 */
ccStatus_t ccGetSliceOutputDim(const ccTensorDescriptor_t xDesc, const ccIntArray_t *begin, const ccIntArray_t *size,
                               int32_t *n, int32_t *c, int32_t *h, int32_t *w, int32_t *realDimCnt);

/**
 * @ingroup dnn
 * @brief slice of a tensor.
 * @param [in] handle  cce handle
 * @param [in] alpha   common scale factor
 * @param [in] xDesc   descriptor of input data
 * @param [in] x       input data in device memory
 * @param [in] begin   begin position of tensor
 * @param [in] size    size to slice
 * @param [in] beta    common scale factor
 * @param [in] yDesc   descriptor of output data
 * @param [in|out] y   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccSliceForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          const ccIntArray_t *begin, const ccIntArray_t *size, const void *beta,
                          const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief gather forward computation
 * @param [in] handle            cce handle
 * @param [in] paramsDesc        descriptor of params tensor
 * @param [in] params            input data in device memory
 * @param [in] indicesDesc       descriptor of indices tensor
 * @param [in] indices           indices data in device memory
 * @param [in] axis              descriptor of roi tensor
 * @param [in] alpha             reserved
 * @param [in] beta              reserved
 * @param [in] outputDesc        descriptor of output tensor
 * @param [out] output           output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccGatherForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t paramsDesc,
                           const void *params, const ccTensorDescriptor_t indicesDesc, const void *indices,
                           const int32_t axis, const void *beta, ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief gather output dim computation, for NC1HWC0
 * @param [in] paramsDesc        descriptor of params tensor
 * @param [in] indicesDesc       descriptor of indices tensor
 * @param [in] axis              descriptor of roi tensor
 * @param [out] n                dim of n
 * @param [out] c                dim of c
 * @param [out] h                dim of h
 * @param [out] w                dim of w
 * @param [out] realDimCnt       real dim count
 * @return ccStatus_t
 */
ccStatus_t ccGetGatherOutputDim(const ccTensorDescriptor_t paramsDesc, const ccTensorDescriptor_t indicesDesc,
                                int32_t axis, int32_t *n, int32_t *c, int32_t *h, int32_t *w, int32_t *realDimCnt);

/**
 * @ingroup dnn
 * @brief gather output dim computation
 * @param [in] paramsDesc        descriptor of params tensor
 * @param [in] indicesDesc       descriptor of indices tensor
 * @param [in] axis              descriptor of roi tensor
 * @param [out] dimCnt           dimcnt of output
 * @param [out] dim              dim of output
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetGatherOutputDim(const ccTensorDescriptor_t paramsDesc, const ccTensorDescriptor_t indicesDesc,
                                int32_t axis, int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief exp forward computation
 * @param [in] handle                    cce handle
 * @param [in] expDesc                   descriptor of expParam
 * @param [in] expParam                  a ternary array
 * @param [in] alpha                     reserved parameter
 * @param [in] xDesc                     descriptor of input tensor
 * @param [in] x                         input data in device memory
 * @param [in] beta                      reserved parameter
 * @param [in] yDesc                     descriptor of output tensor
 * @param [out] y                        output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccExpForward(ccHandle_t handle, const ccExpDescriptor_t expDesc, const void *expParam, const void *alpha,
                        const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                        const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief expm1 forward:
 *          data type only support float float16 and double
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x            input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccExpm1Forward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief log1p forward:
 *          data type only support float float16 and double
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x            input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccLog1pForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief init descriptor for parameter of exp function
 * @param [in|out] powDesc   descriptor of tensor
 * @param [in] dataType   data type in device
 * @param [in] paramCnt   number of parameters
 * @return ccStatus_t
 */
ccStatus_t ccSetExpDescriptor(ccExpDescriptor_t expDesc, ccDataType_t dataType, uint32_t paramCnt);

/**
 * @ingroup dnn
 * @brief exp forward computation
 * @param [in] handle                    cce handle
 * @param [in] logDesc                   descriptor of logParam
 * @param [in] logParam                  a ternary array
 * @param [in] alpha                     reserved parameter
 * @param [in] xDesc                     descriptor of input tensor
 * @param [in] x                         input data in device memory
 * @param [in] beta                      reserved parameter
 * @param [in] yDesc                     descriptor of output tensor
 * @param [in] y                         output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccLogForward(ccHandle_t handle, const ccLogDescriptor_t logDesc, const void *logParam, const void *alpha,
                        const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                        const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief init descriptor for parameter of log function
 * @param [in|out] logDesc   descriptor of tensor
 * @param [in] dataType   data type in device
 * @param [in] paramCnt   number of parameters
 * @return ccStatus_t
 */
ccStatus_t ccSetLogDescriptor(ccLogDescriptor_t logDesc, ccDataType_t dataType, uint32_t paramCnt);

/**
 * @ingroup dnn
 * @brief pow forward computation
 * @param [in] handle                    cce handle
 * @param [in] powDesc                   descriptor of logParam
 * @param [in] powParam                  a ternary array
 * @param [in] alpha                     reserved parameter
 * @param [in] xDesc                     descriptor of input tensor
 * @param [in] x                         input data in device memory
 * @param [in] beta                      reserved parameter
 * @param [in] yDesc                     descriptor of input tensor
 * @param [in] y                         input data in device memory
 * @param [in] zDesc                     descriptor of output tensor
 * @param [out] z                        output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccPowForward(ccHandle_t handle, const ccPowDescriptor_t powDesc, const void *powParam, const void *alpha,
                        const ccTensorDescriptor_t xDesc, const void *x, const ccTensorDescriptor_t yDesc,
                        const void *y, const void *beta, const ccTensorDescriptor_t zDesc, void *z);

/**
 * @brief init descriptor for parameter of pow function
 * @param [in|out] powDesc   descriptor of tensor
 * @param [in] dataType   data type in device
 * @param [in] paramCnt   number of parameters
 * @return ccStatus_t
 */
ccStatus_t ccSetPowDescriptor(ccPowDescriptor_t powDesc, ccDataType_t dataType, uint32_t paramCnt);

/**
 * @ingroup dnn
 * @brief non max suppression forward.
 * @param [in] handle                 cce handle
 * @param [in] nonmaxParaDesc         descriptor of para
 * @param [in] nonmaxPara             input para in host memory
 * @param [in] maxoutputsizex         input para in host memory
 * @param [in] alpha                  common scale factor
 * @param [in] boxesDesc              descriptor of input data boxesDesc
 * @param [in] boxes                  input data boxes in device memory
 * @param [in] scoresDesc             descriptor of input data boxesDesc
 * @param [in] scores                 input data scores in device memory
 * @param [in] workSpaceSizeInBytes   workspace size
 * @param [in] workSpace              input workspace in device memory
 * @param [in] beta                   common scale factor
 * @param [in] outputDesc             descriptor of output data
 * @param [in|out] output             output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccNonMaxSuppressionForward(ccHandle_t handle, const ccNonMaxSuppressionDescriptor_t nonmaxParaDesc,
                                      const void *nonmaxPara, const int *maxoutputsize, const void *alpha,
                                      const ccTensorDescriptor_t boxesDesc, const void *boxes,
                                      const ccTensorDescriptor_t scoresDesc, const void *scores,
                                      const uint32_t workSpaceSizeInBytes, void *workSpace, const void *beta,
                                      const ccTensorDescriptor_t outputDesc, void *output);
/**
 * @brief init descriptor for parameter of NonMaxSuppression function
 * @param [in|out] powDesc   descriptor of tensor
 * @param [in] dataType   data type in device
 * @param [in] paramCnt   number of parameters
 * @return ccStatus_t
 */
ccStatus_t ccSetNonMaxSuppressionDescriptor(ccNonMaxSuppressionDescriptor_t nonMaxSuppressionDesc,
                                            ccDataType_t dataType, uint32_t paramCnt);

/**
 * @ingroup dnn
 * @brief get the output dimension info of resizeBilinear op.
 * @param [in] xDesc                    descriptor of input data
 * @param [in] resizeBilinearDesc       descriptor of resize_bilinear operator
 * @param [out] dimCnt
 * @param [out] dim[]                   dim of output
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetResizeBilinearOutputDim(const ccTensorDescriptor_t xDesc,
                                        const ccResizeBilinearDescriptor_t resizeBilinearDesc, int32_t *dimCnt,
                                        int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief get the output dimension info of interp op.
 * @param [in] xDesc                    descriptor of input data
 * @param [in] resizeBilinearDesc       descriptor of resize_bilinear operator
 * @param [out] dimCnt
 * @param [out] dim[]                   dim of output
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetInterpOutputDim(const ccTensorDescriptor_t xDesc, const ccResizeBilinearDescriptor_t resizeBilinearDesc,
                                int32_t *dimCnt, int32_t dim[], int32_t dimLen);
/**
 * @ingroup dnn
 * @brief resize bilinear forward for t network.
 * @param [in] handle    cce handle
 * @param [in] resizeBilinearDesc   descriptor of resize_bilinear operator
 * @param [in] alpha     common scale factor
 * @param [in] xDesc     descriptor of input data
 * @param [in] x         input data in device memory
 * @param [in] beta      common scale factor
 * @param [in] yDesc     descriptor of output data
 * @param [in|out] y     output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccResizeBilinearForward(ccHandle_t handle, const ccResizeBilinearDescriptor_t resizeBilinearDesc,
                                   const void *alpha, const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                   const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief resize bilinear forward for c network.
 * @param [in] handle    cce handle
 * @param [in] resizeBilinearDesc   descriptor of resize_bilinear operator
 * @param [in] alpha     common scale factor
 * @param [in] xDesc     descriptor of input data
 * @param [in] x         input data in device memory
 * @param [in] beta      common scale factor
 * @param [in] yDesc     descriptor of output data
 * @param [in|out] y     output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccInterpForward(ccHandle_t handle, const ccResizeBilinearDescriptor_t resizeBilinearDesc, const void *alpha,
                           const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                           const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief create descriptor of ResizeBilinear
 * @param [in|out] resizeBilinearDesc   point to descriptor of resizeBilinear attr
 * @return ccStatus_t
 */
ccStatus_t ccCreateResizeBilinearDescriptor(ccResizeBilinearDescriptor_t *resizeBilinearDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of Interp
 * @param [in|out] resizeBilinearDesc   point to descriptor of resizeBilinear attr
 * @return ccStatus_t
 */
ccStatus_t ccDestroyResizeBilinearDescriptor(ccResizeBilinearDescriptor_t *resizeBilinearDesc);

/**
 * @ingroup dnn
 * @brief set descriptor of resizeBilinear.
 * @param [in|out] resizeBilinearDesc   descriptor of resize_bilinear operator
 * @param [in] resizeOutputDimMode      way to decide output dimensions
 * @param [in] alignCorners             whether the centers of input and output are aligned
 * @param [in] zoom_factor              zoom factor
 * @param [in] shrink_factor            shrink factor
 * @param [in] height                   height of output
 * @param [in] width                    width of output
 * @param [in] pad_begin                padding at begin of input
 * @param [in] pad_end                  padding at end of input
 * @return ccStatus_t
 */
ccStatus_t ccSetResizeBilinearDescriptor(ccResizeBilinearDescriptor_t resizeBilinearDesc,
                                         ccResizeOutputDimMode_t resizeOutputDimMode, bool alignCorners,
                                         int32_t zoom_factor, int32_t shrink_factor, int32_t height, int32_t width,
                                         int32_t pad_begin, int32_t pad_end);

/**
 * @ingroup dnn
 * @brief fill forward computation
 * @param [in] handle                  cce handle
 * @param [in] fillParamDesc           descriptor of fill parameter
 * @param [in] alpha                   reserved
 * @param [in] givenDesc               descriptor of given tensor
 * @param [in] givenData               given data in device memory
 * @param [in] workspace               space for fill algorithm
 * @param [in] workSpaceSizeInBytes    space size in byte
 * @param [in] beta                    reserved
 * @param [in] outputDesc              descriptor of output tensor
 * @param [out] output                 output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccFillForward(ccHandle_t handle, const ccFillParamDescriptor_t fillParamDesc, const void *alpha,
                         const ccTensorDescriptor_t givenDesc, const void *givenData, const void *workspace,
                         const uint32_t workSpaceSizeInBytes, const void *beta, const ccTensorDescriptor_t outputDesc,
                         void *output);

/**
 * @ingroup dnn
 *[ccGetFillWorkspaceSize]
 *@param fillType         [fill type]
 *@param givenDesc        [given tensor descriptor]
 *@param xDesc            [input tensor descriptor]
 *@param sizeInBytes      [output size]
 *@return ccStatus_t      [status]
 */
ccStatus_t ccGetFillWorkspaceSize(const ccFillOpType_t fillType, const ccTensorDescriptor_t xDesc,
                                  uint32_t *sizeInBytes);

/**
 *[ccCast]
 *@param handle     [cce handler]
 *@param alpha       [alpha]
 *@param xDesc      [tensor Description of tensor x]
 *@param x             [input tensor x]
 *@param beta         [beta
 *@param yDesc      [tensor Description of tensor y]
 *@param y             [output tensor y]
 *@return ccStatus_t  [status]
 */
ccStatus_t ccCast(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                  const void *beta, const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief round forward:
 *          data type only support float float16 and int32
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x            input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccRoundForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief rint forward:
 *          data type only support float float16
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x            input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccRintForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                         const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief sqrt forward:
 *          data type only support float float16
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x            input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccSqrtForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                         const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 *[ccCast]
 *@param filterSrcInfo     [cce filtersrc descriptor]
 *@param filterSrc       [filterSrc address]
 *@param filterDstInfo      [cce filterdst descriptor]
 *@param filterDst             [filterdst address]
 *@param group         [group]
 *@param ySizeInBytes      [fraczfilter size]
 *@param outputDataType            [datatype]
 *@return ccStatus_t  [status]
 */
ccStatus_t ccTransGroupConvFilterInt8(ccFilterDescriptor_t filterSrcInfo, const void *filterSrc,
                                      ccFilterDescriptor_t filterDstInfo, void *filterDst, uint32_t group,
                                      uint32_t ySizeInBytes, ccDataType_t outputDataType);

/**
 *[ccGetConcatOutputDim]
 *@param xDesc[]     [input tensor descriptor]
 *@param axis        [concat axis]
 *@param inputNum    [input tensor numbers]
 *@param dim[]       [output dim]
 *@param [in| dimlen        length of dim
 *@return ccStatus_t [status]
 */
ccStatus_t ccGetConcatOutputDim(const ccTensorDescriptor_t xDesc[], int32_t axis, int32_t inputNum, int32_t *dimCnt,
                                int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief get the output dimension info of reduce.
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] axis             The dimensions to reduce
 * @param [in] keepDims         If true, retains reduced dimensions with length 1.
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetReduceOutputDim(const ccTensorDescriptor_t xDesc, const ccIntArray_t *axis, bool keepDims,
                                int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief reduce sum forward computation
 * @param [in] handle          cce handle
 * @param [in] axis            The dimensions to reduce
 * @param [in] keepDims        If true, retains reduced dimensions with length 1.
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [in|out] output      output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReduceSumForward(ccHandle_t handle, const ccIntArray_t *axis, bool keepDims, const void *alpha,
                              const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                              const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief reduce max forward computation
 * @param [in] handle          cce handle
 * @param [in] axis            The dimensions to reduce
 * @param [in] keepDims        If true, retains reduced dimensions with length 1.
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [in|out] output      output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReduceMaxForward(ccHandle_t handle, const ccIntArray_t *axis, bool keepDims, const void *alpha,
                              const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                              const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief reduce min forward computation
 * @param [in] handle          cce handle
 * @param [in] axis            The dimensions to reduce
 * @param [in] keepDims        If true, retains reduced dimensions with length 1.
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [in|out] output      output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReduceMinForward(ccHandle_t handle, const ccIntArray_t *axis, bool keepDims, const void *alpha,
                              const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                              const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief reduce mean forward computation
 * @param [in] handle          cce handle
 * @param [in] axis            The dimensions to reduce
 * @param [in] keepDims        If true, retains reduced dimensions with length 1.
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [in|out] output      output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReduceMeanForward(ccHandle_t handle, const ccIntArray_t *axis, bool keepDims, const void *alpha,
                               const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                               const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief reduce prod forward computation
 * @param [in] handle          cce handle
 * @param [in] axis            The dimensions to reduce
 * @param [in] keepDims        If true, retains reduced dimensions with length 1.
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [in|out] output      output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReduceProdForward(ccHandle_t handle, const ccIntArray_t *axis, bool keepDims, const void *alpha,
                               const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                               const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief reduce all forward computation
 * @param [in] handle          cce handle
 * @param [in] axis            The dimensions to reduce
 * @param [in] keepDims        If true, retains reduced dimensions with length 1.
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [in|out] output      output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReduceAllForward(ccHandle_t handle, const ccIntArray_t *axis, bool keepDims, const void *alpha,
                              const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                              const ccTensorDescriptor_t outputDesc, void *output);

/**
 *@brief print times stats
 *@return ccStatus_t  [status]
 */
ccStatus_t ccPrintTimeStat();

/**
 * @ingroup dnn
 * @brief reduce abs sum forward computation
 * @param [in] handle          cce handle
 * @param [in] axis            The dimensions to reduce
 * @param [in] keepDims        If true, retains reduced dimensions with length 1.
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [in|out] output      output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReduceAbsSumForward(ccHandle_t handle, const ccIntArray_t *axis, const bool keepDims, const void *alpha,
                                 const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                 const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief reduce square sum forward computation
 * @param [in] handle          cce handle
 * @param [in] axis            The dimensions to reduce
 * @param [in] keepDims        If true, retains reduced dimensions with length 1.
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [in|out] output      output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReduceSquareSumForward(ccHandle_t handle, const ccIntArray_t *axis, const bool keepDims, const void *alpha,
                                    const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                    const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get the output dimension info of crop and resize
 * @param [in] imageDesc             descriptor of images
 * @param [in] boxesDesc             descriptor of boxes
 * @param [in] boxidxDesc            descriptor of boxidx
 * @param [in] resizeHeight          resize height
 * @param [in] resizeWidth           resize width
 * @param [out] dimCnt               dimcnt of output
 * @param [out] dim                  dim of output
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetCropAndResizeOutputDim(const ccTensorDescriptor_t imageDesc, const ccTensorDescriptor_t boxesDesc,
                                       const ccTensorDescriptor_t boxidxDesc, const int32_t resizeHeight,
                                       const int32_t resizeWidth, int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief crop and resize forward.
 * @param [in] handle                cce handle
 * @param [in] alpha                 common scale factor
 * @param [in] imageDesc             descriptor of images
 * @param [in] image                 input data in device memory
 * @param [in] boxesDesc             descriptor of boxes
 * @param [in] boxes                 input data in device memory
 * @param [in] boxidxDesc            descriptor of boxidx
 * @param [in] boxidx                input data in device memory
 * @param [in] method                enum of resize method
 * @param [in] extrapolationValue    Value used for extrapolation, when applicable
 * @param [in] beta                  common scale factor
 * @param [in] outputDesc            descriptor of output data
 * @param [out] output               output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccCropAndResizeForward(ccHandle_t handle, const ccResizeMethod_t method, const float extrapolationValue,
                                  const void *alpha, const ccTensorDescriptor_t imageDesc, const void *image,
                                  const ccTensorDescriptor_t boxesDesc, const void *boxes,
                                  const ccTensorDescriptor_t boxidxDesc, const void *boxidx, const void *beta,
                                  const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief select forward computation
 * @param [in] handle            cce handle
 * @param [in] alpha             reserved
 * @param [in] condDesc          descriptor of cond tensor
 * @param [in] cond              cond data in device memory
 * @param [in] xDesc             descriptor of x tensor
 * @param [in] x                 x data in device memory
 * @param [in] yDesc             descriptor of y tensor
 * @param [in] y                 y data in device memory
 * @param [in] beta              reserved
 * @param [in] outputDesc        descriptor of output tensor
 * @param [out] output           output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccSelect(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t condDesc, const void *cond,
                    const ccTensorDescriptor_t xDesc, const void *x, const ccTensorDescriptor_t yDesc, const void *y,
                    const void *beta, const ccTensorDescriptor_t outDesc, void *out);

/**
 * @ingroup dnn
 * @brief get the output dimension info of where
 * @param [in] xDesc            descriptor of input tensor
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @return ccStatus_t
 */
ccStatus_t ccGetWhereOutputDim(const ccTensorDescriptor_t xDesc, int32_t *dimCnt, int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief where forward computation
 * @param [in] handle            cce handle
 * @param [in] alpha             reserved
 * @param [in] condDesc          descriptor of cond tensor
 * @param [in] cond              cond data in device memory
 * @param [in] xDesc             descriptor of x tensor
 * @param [in] x                 x data in device memory
 * @param [in] yDesc             descriptor of y tensor
 * @param [out] y                y data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccWhere(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                   const void *beta, const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief reverse forward.
 * @param [in] handle       cce handle
 * @param [in] axis         dim that need reverse
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x            input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReverseForward(ccHandle_t handle, const ccIntArray_t *axis, const void *alpha,
                            const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                            const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief floor forward:
 *          data type only support float float16
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x            input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccFloorForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief ceil forward:
 *          data type only support float float16
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x            input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccCeilForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                         const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get the output dimension info of truncate mod
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] yDesc            descriptor of input tensor
 * @param [out] dimCnt        [dim count of the output tensor]
 * @param [out] dim[]         [shape of the output tensor]
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetTruncatemodOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc,
                                     int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief truncate mod forward computation
 * @param [in] handle          cce handle
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] yDesc           descriptor of input tensor
 * @param [in] y               input data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [out] output         output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccTruncatemodForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                                const ccTensorDescriptor_t outputDesc, void *output);
/**
 * @ingroup dnn
 * @brief Spatial Pyramid Pooling
 * @param [in] handle                cce handle
 * @param [in] alpha                 reserved
 * @param [in] xDesc                 descriptor of input tensor
 * @param [in] x                     input data in device memory
 * @param [in] workspace             temp workspace
 * @param [in] workspaceSizeInBytes  temp workspace size
 * @param [in] pyramidHeight         pyramid height
 * @param [in] poolingMode           pooling mode
 * @param [in] beta                  reserved
 * @param [in] outputDesc            descriptor of output tensor
 * @param [out] output               output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccSPPForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                        void *workspace, const uint32_t workspaceSizeInBytes, const uint32_t pyramidHeight,
                        const ccPoolingMode_t poolingMode, const void *beta, const ccTensorDescriptor_t outputDesc,
                        void *output);
/**
 * @ingroup dnn
 * @brief Get Spatial Pyramid Pooling output dim
 * @param [in] xDesc                 descriptor of input tensor
 * @param [in] pyramidHeight         pyramid height
 * @param [in] dimLen                length of dim
 * @param [out] dimCnt               output tensor dim cnt
 * @param [out] dim                  output tensor dim
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetSPPOutputDim(const ccTensorDescriptor_t xDesc, const uint32_t pyramidHeight, int32_t *dimCnt,
                             int32_t dim[], const int32_t dimLen);
/**
 * @ingroup dnn
 * @brief Get Spatial Pyramid Pooling workspace size
 * @param [in] xDesc                 descriptor of input tensor
 * @param [in] pyramidHeight         pyramid height
 * @param [out] workspaceSizeInBytes workspace size
 * @return ccStatus_t
 */
ccStatus_t ccGetSPPWorkspaceSize(const ccTensorDescriptor_t xDesc, const uint32_t pyramidHeight,
                                 uint32_t *workspaceSizeInBytes);

/**
 * @ingroup dnn
 * @brief BNLL forward computation
 * @param [in] handle           cce handle
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] x                input data in device memory
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccBNLLForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                         const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief bias forward.
 * @param [in] handle       cce handle
 * @param [in] axis            axis
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data x
 * @param [in] x            input data x in device memory
 * @param [in] biasDesc        descriptor of input data bias
 * @param [in] bias            input data bias in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccBiasForward(ccHandle_t handle, const int axis, const void *alpha, const ccTensorDescriptor_t xDesc,
                         const void *x, const ccTensorDescriptor_t biasDesc, const void *bias, const void *beta,
                         const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief threshold forward computation
 * @param [in] handle           cce handle
 * @param [in] threshold        threshold
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] x                input data in device memory
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccThresholdForward(ccHandle_t handle, const void *threshold, const void *alpha,
                              const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                              const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief shufflechannel forward.
 * @param [in] handle    cce handle
 * @param [in] alpha     common scale factor
 * @param [in] group     number of groups
 * @param [in] xDesc     descriptor of input data
 * @param [in] x         input data in device memory
 * @param [in] beta      common scale factor
 * @param [in] outputDesc     descriptor of output data
 * @param [in|out] output     output data in device memory
 * @return ccStatus_t
 */
// TODO AICPU: please add shufflechannel custom params and comment
ccStatus_t ccShuffleChannelForward(ccHandle_t handle, const void *alpha, uint32_t group,
                                   const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                   const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief mvn forward.
 * @param [in] handle               cce handle
 * @param [in] acrossChannel        across channel. true: across, false: not
 * @param [in] normalizeVariance    normalizeVariance. true: normalizeVariance, false: not
 * @param [in] alpha                common scale factor
 * @param [in] xDesc                descriptor of input data
 * @param [in] x                    input data in device memory
 * @param [in] beta                 common scale factor
 * @param [in] outputDesc           descriptor of output data
 * @param [in|out] output           output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccMVNForward(ccHandle_t handle, bool acrossChannel, bool normalizeVariance, const void *alpha,
                        const ccTensorDescriptor_t xDesc, const void *x, void *workSpace, uint32_t workSpaceSizeInBytes,
                        const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get the workspace size of mvn
 * @param [in] xDesc                descriptor of input data
 * @param [in] acrossChannel        across channel. true: across, false: not
 * @param [in|out] sizeInBytes      Workspace size need for whole computation
 */
ccStatus_t ccGetMVNWorkspaceSize(const ccTensorDescriptor_t xDesc, bool acrossChannel, uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief heatmap2coord forward output is hotspot value and corresponding coordinates
 * @param [in] handle        cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x        input data in device memory
 * @param [in] coordh       calibration high
 * @param [in] coordw       calibration wide
 * @param [in] beta        common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccHeatmap2coordForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                  int32_t coordh, int32_t coordw, const void *beta,
                                  const ccTensorDescriptor_t outputDesc, void *output);
/**
 * @ingroup dnn
 * @brief get the output dimension info of heatmap2coord
 * @param [in] xDesc           descriptor of input tensor
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim           arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetHeatmap2coordOutputDim(const ccTensorDescriptor_t xDesc, int32_t *dimCnt, int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief swish forward.
 * @param [in] handle           cce handle
 * @param [in] scale            param of swish function, y = x / (1 + sigmoid(scale * x))
 * @param [in] alpha            common scale factor
 * @param [in] xDesc            descriptor of input data
 * @param [in] x                input data in device memory
 * @param [in] beta             common scale factor
 * @param [in] outputDesc       descriptor of output data
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */

ccStatus_t ccSwishForward(ccHandle_t handle, const float scale, const void *alpha, const ccTensorDescriptor_t xDesc,
                          const void *x, const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

ccStatus_t ccTeForward(ccHandle_t handle, const void *stubFunc, uint32_t coreDim, const void *args, uint32_t argsSize,
                       const rtL2Ctrl_t *l2ctrl, int32_t inputNum, const ccTensorDescriptor_t xDesc[], const void *x[],
                       int32_t outputNum, const ccTensorDescriptor_t yDesc[], void *y[], bool isAiCore);

#ifndef DAVINCI_LITE
ccStatus_t ccAiCpuCustomizeForward(ccHandle_t handle, aicpu_run_func stubFunc, opTensor_t *xOpDesc[], void *x[],
                                   int32_t inputNum, opTensor_t *yOpDesc[], void *y[], void *op_attr_handle,
                                   int32_t outputNum, const ccTensorDescriptor_t xDesc[],
                                   const ccTensorDescriptor_t yDesc[], const void *op_attr_str, uint32_t op_attr_size);
#endif
/**
 * @ingroup dnn
 * @brief embedding lookup forward.
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data x
 * @param [in] x            input data x in device memory
 * @param [in] idxDesc        descriptor of input data idx
 * @param [in] idx            input data idx in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccEmbeddingLookupForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc,
                                    const void *x, const ccTensorDescriptor_t idxDesc, const void *idx,
                                    const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup
 * @brief embedding lookup forward.
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] inputNum   inputNum
 * @param [in] xDesc[]        descriptor array of input data x
 * @param [in] x[]            input data x array in device memory
 * @param [in] workSpace    workSpace addr
 * @param [in] workSpaceSizeInBytes    workSpace size
 * @param [in] idxDesc        descriptor of input data idx
 * @param [in] idx            input data idx in device memory
 * @param [in] partitionStrategy  partitionStrategy
 * @param [in] maxNorm            addr of maxNorm
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccEmbeddingLookupForward(ccHandle_t handle, const void *alpha, const int32_t inputNum,
                                    const ccTensorDescriptor_t xDesc[], const void *x[], void *workSpace,
                                    const uint32_t workSpaceSizeInBytes, const ccTensorDescriptor_t idxDesc,
                                    const void *idx, ccPartitionStrategy_t partitionStrategy, const void *maxNorm,
                                    const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 *[ccGetEmbeddingLookupOutputDim]
 *@param inputNum    [input tensor numbers]
 *@param xDesc[]     [input tensor descriptor]
 *@param idxDesc     [idx tensor descriptor]
 *@param dimCnt      [output dim count]
 *@param dim[]       [output dim]
 *@param [in| dimlen        length of dim
 *@return ccStatus_t [status]
 */
ccStatus_t ccGetEmbeddingLookupOutputDim(const int32_t inputNum, const ccTensorDescriptor_t xDesc[],
                                         const ccTensorDescriptor_t idxDesc, int32_t *dimCnt, int32_t dim[],
                                         int32_t dimLen);

/**
 * @ingroup dnn
 *[ccGetEmbeddingLookupWorkspaceSize]
 *@param inputNum    [input tensor numbers]
 *@param idxDesc      [input tensor descriptor]
 *@param isMaxNormExist      [isMaxNormExist]
 *@param sizeInBytes      [output size]
 *@return ccStatus_t [status]
 */
ccStatus_t ccGetEmbeddingLookupWorkspaceSize(const int32_t inputNum, const ccTensorDescriptor_t idxDesc,
                                             const bool isMaxNormExist, uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief check if it is the first layer of resnet50 and semecefc
 * @param [in] tensorDesc           descriptor of input tensor.
 * @param [in] convDesc             conv descriptor.
 * @param [in] filterDesc           descriptor of weight tensor.
 * @return ccStatus_t
 */
ccStatus_t c04DescParamCheck(const ccTensorDescriptor_t tensorDesc, const ccConvolutionDescriptor_t convDesc,
                             const ccFilterDescriptor_t filterDesc);

#ifndef DAVINCI_LITE
/**
 * @ingroup dnn
 * @brief convolution forward computation
 * @param [in] handle   cce handle
 * @param [in] convDesc   descriptor of convolution operator
 * @param [in] alpha   scaling factors
 * @param [in] beta   scaling factors
 * @param [in] xDesc  x descriptor of input tensor
 * @param [in] x   x data in device memory
 * @param [in] dyDesc   descriptor of dy
 * @param [in] dy   dy data in device memory
 * @param [in] dwDesc   descriptor of dwDesc
 * @param [out] dw   dw data in device memory
 * @param [in] algo   algorithm of convolution forward
 * @param [in] workSpace   temp space, maybe NULL if no need temp space
 * @param [in] workSpaceSizeInBytes   sizeof workspace
 * @return ccStatus_t
 */
ccStatus_t ccConvolutionBackwardFilter(ccHandle_t handle, const ccConvolutionDescriptor_t convDesc, void *alpha,
                                       void *beta, const ccTensorDescriptor_t xDesc, const void *x,
                                       const ccTensorDescriptor_t dyDesc, const void *dy,
                                       const ccFilterDescriptor_t dwDesc, void *dw, ccConvolutionBwdAlgo_t algo,
                                       void *workSpace, uint32_t workSpaceSizeInBytes);
#endif

/**
 * @ingroup dnn
 * @brief get the temp space size of convolution forward computation, maybe no need temp space
 * @param [in] handle   cce handle
 * @param [in] dyDesc   descriptor of input tensor dy
 * @param [in] convDesc   descriptor of convolution operator
 * @param [in] xDesc   descriptor of input tensor
 * @param [in] dwDesc   descriptor of filter
 * @param [in] algo   algorithm of convolution forward
 * @param [in|out] sizeInBytes   temp space size need for specified algorithm
 * @return ccStatus_t
 */
ccStatus_t ccGetConvolutionBackwardFilterWorkspaceSize(ccHandle_t handle, const ccTensorDescriptor_t dyDesc,
                                                       const ccConvolutionDescriptor_t convDesc,
                                                       const ccTensorDescriptor_t xDesc,
                                                       const ccFilterDescriptor_t dwDesc, ccConvolutionBwdAlgo_t algo,
                                                       uint32_t *sizeInBytes);

#ifndef DAVINCI_LITE
ccStatus_t ccBatchNormalizationBackward(ccHandle_t handle, ccBatchNormMode_t mode, const void *alphaDataDiff,
                                        const void *betaDataDiff, const void *alphaParamDiff, const void *betaParamDiff,
                                        const ccTensorDescriptor_t xDesc, const void *x,
                                        const ccTensorDescriptor_t dyDesc, const void *dy,
                                        const ccTensorDescriptor_t dxDesc, void *dx,
                                        const ccTensorDescriptor_t bnScaleBiasDiffDesc, const void *bnScale,
                                        void *resultBnScaleDiff, void *resultBnBiasDiff, const void *workSpace,
                                        const uint32_t workSpaceSizeInBytes, double epsilon, const void *SaveMean,
                                        const void *SaveInvVariance);
#endif

ccStatus_t ccGetBatchNormalizationBackwardWorkspaceSize(ccHandle_t handle, ccBatchNormMode_t mode,
                                                        ccTensorDescriptor_t xDesc, ccTensorDescriptor_t dyDesc,
                                                        ccTensorDescriptor_t dxDesc,
                                                        ccTensorDescriptor_t bnScaleBiasDesc, uint32_t *sizeInBytes);

#ifndef DAVINCI_LITE
ccStatus_t ccBatchNormalizationForwardTraining(ccHandle_t handle, ccBatchNormMode_t mode, const void *alpha,
                                               const void *beta, const ccTensorDescriptor_t xDesc, const void *x,
                                               const ccTensorDescriptor_t yDesc, void *y,
                                               const ccTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
                                               const void *bnBias, double exponentialAverageFactor,
                                               void *resultRunningMean, void *resultRunningVariance, void *workSpace,
                                               uint32_t workSpaceSizeInBytes, double epsilon, void *resultSaveMean,
                                               void *resultSaveInvVariance, const bool isTraining);
#endif

ccStatus_t ccGetBatchNormalizationForwardTrainingWorkspaceSize(ccHandle_t handle, ccBatchNormMode_t mode,
                                                               ccTensorDescriptor_t xDesc, ccTensorDescriptor_t yDesc,
                                                               const ccTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                                               uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief generate an random normal Tensor use given on/off scale.
 * @param [in] handle        Stream handle.
 * @param [in] alpha         reserved.
 * @param [in] meanDesc      Mean description of one-hot position.
 * @param [in] mean          Data pointer of mean.
 * @param [in] scaleDesc     On/off scale description.
 * @param [in] scale         Data pointer of on/off scale.
 * @param [in] seed          random seed used to generate random number
 * @param [in] seed2         random seed used to generate random number
 * @param [in] beta          reserved.
 * @param [in] outputDesc    Description of the generated one-hot tensor.
 * @param [output] output    Data pointer of output.
 * @return ccStatus_t
 */
ccStatus_t ccRandomNormalForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t meanDesc,
                                 const void *mean, const ccTensorDescriptor_t scaleDesc, const void *scale,
                                 const int64_t seed1, const int64_t seed2, const void *beta,
                                 const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief generate random uniform tensor.
 * @param [in] handle        Stream handle.
 * @param [in] alpha         reserved.
 * @param [in] minvalDesc    Mean description of one-hot position.
 * @param [in] minval        Data pointer of mean.
 * @param [in] maxvalDesc    On/off scale description.
 * @param [in] maxval        Data pointer of on/off scale.
 * @param [in] seed          random seed used to generate random number
 * @param [in] seed2         random seed used to generate random number
 * @param [in] beta          reserved.
 * @param [in] outputDesc    Description of the generated one-hot tensor.
 * @param [output] output    Data pointer of output.
 * @return ccStatus_t
 */
ccStatus_t ccRandomUniformForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t minvalDesc,
                                  const void *minval, const ccTensorDescriptor_t maxvalDesc, const void *maxval,
                                  const int64_t seed1, const int64_t seed2, const void *beta,
                                  const ccTensorDescriptor_t outputDesc, void *output);

/**^M
 * @ingroup dnn^M\r	10932
 * @brief generate BatchMatMul tensor.^M\r	10933
 * @param [in] handle        Stream handle.^M\r	10934
 * @param [in] alpha         reserved.^M\r	10935
 * @param [in] xDesc         tensorA Desc.^M\r	10936
 * @param [in] x             Data pointer of tensorA.^M\r	10937
 * @param [in] yDesc         tensorB Desc.^M\r	10938
 * @param [in] y             Data pointer of tensorB.^M\r	10939
 * @param [in] beta          reserved.^M\r	10940
 * @param [in] adj_x         tensorA transpose flag^M\r	10941
 * @param [in] adj_y         tensorB transpose flag^M\r	10942
 * @param [in] outpDesc      Description of the tensor output .^M\r	10943
 * @param [output] out       Data pointer of output.^M\r	10944
 * @return ccStatus_t^M
 */
ccStatus_t ccBatchMatMulForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                const ccTensorDescriptor_t yDesc, const void *y, const void *beta, const bool adj_x,
                                const bool adj_y, const ccTensorDescriptor_t outDesc, void *out);

ccStatus_t ccGetBatchMatMulOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, bool adj_x,
                                     bool adj_y, int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief generator conv int8 all offset factor
 * @param [in] para   the struct for scale and offset of input, filter and output
 * @param [in|out] offsetW   offset of filter
 * @param [in|out] offsetPad   offset of input
 * @param [in|out] scaledQrq   scale computing result of input , filter and output
 * @param [in|out] nextoffsetq   offset of output
 * @return ccStatus_t
 */
ccStatus_t ccGenQuantAllOffsetFactor(const ccQuantAllOffsetPara_t *para, uint8_t &offsetW, uint8_t &offsetPad,
                                     uint16_t &scaledQrq, uint16_t &nextoffsetq);

/**
 * @ingroup dnn
 * @brief get conv int8 all offset fracZ size
 * @param [in] filterDesc   descriptor of filter tensor
 * @param [in|out] conv int8 all offset fracZ size
 * @param [in] groupNum   group conv num
 * @return ccStatus_t
 */
ccStatus_t ccSetGroupConvScene(const ccFilterDescriptor_t tensorDesc, ccConvolutionDescriptor_t convDesc);

ccStatus_t ccGetInt8AllOffsetFilterFracZSizeInBytes(const ccFilterDescriptor_t filterSrcDesc,
                                                    const ccFilterDescriptor_t filterDesc, uint32_t &size,
                                                    uint32_t groupNum);

/**
 * @ingroup dnn
 * @brief transform filter in conv int8 all offset scene
 * @param [in] filterSrcInfo    descriptor of filter tensor before fracZ transform
 * @param [in] filterSrc        filter addr before fracZ transform
 * @param [in] filterDstInfo   descriptor of filter tensor after fracZ transform
 * @param [in] filterDst   filter addr after fracZ transform
 * @param [in] quantPara   the struct for scale and offset of input, filter and output
 * @param [in] ySizeInBytes   filter size after fracZ transform
 * @param [in|out] outputDataType   output data type
 * @param [in] groupNum   group conv num
 * @return ccStatus_t
 */
ccStatus_t ccTransFilterInt8AllOffset(ccFilterDescriptor_t filterSrcInfo, const void *filterSrc,
                                      ccFilterDescriptor_t filterDstInfo, void *filterDst,
                                      const ccQuantAllOffsetPara_t *quantPara, uint32_t ySizeInBytes,
                                      ccDataType_t outputDataType, uint32_t groupNum);

/**
 * @ingroup dnn
 * @brief transform bias in conv int8 all offset scene
 * @param [in] filterDesc    descriptor of filter tensor
 * @param [in] biasDesc     descriptor of bias tensor
 * @param [in] quantPara   the struct for scale and offset of input, filter and output
 * @param [in] w      filter addr
 * @param [in] bias   bias addr
 * @return ccStatus_t
 */
ccStatus_t ccTransInt8AllOffsetBias(const ccFilterDescriptor_t filterDesc, const ccTensorDescriptor_t biasDesc,
                                    const ccQuantAllOffsetPara_t *quantPara, const void *w, const void *bias);

/**
 * @ingroup dnn
 * @get dequantize
 * @param [in] handle  handle id
 * @param [in] alpha  alpha addr
 * @param [in] xDesc the input Desc  descriptor
 * @param [in] x   x data addr
 * @param [in] beta beta data addr
 * @param [in] yDesc the output Desc  descriptor
 * @param [in] y   y data addr
 * @return ccStatus_t
 */
ccStatus_t ccDequantizeCoreForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc,
                                   const void *x, const void *beta, const ccTensorDescriptor_t yDesc, void *y);
/**
 * @ingroup dnn
 * @get quantize
 * @param [in] handle  handle id
 * @param [in] alpha  alpha addr
 * @param [in] xDesc the input Desc  descriptor
 * @param [in] x   x data addr
 * @param [in] beta beta data addr
 * @param [in] yDesc the output Desc  descriptor
 * @param [in] y   y data addr
 * @return ccStatus_t
 */
ccStatus_t ccQuantizeCoreForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                 const void *beta, const ccTensorDescriptor_t yDesc, void *y);

#ifndef DAVINCI_LITE
ccStatus_t ccActivationBackward(ccHandle_t handle, const ccActivationDescriptor_t activationDesc, const void *alpha,
                                const ccTensorDescriptor_t dyDesc, const void *dy, const ccTensorDescriptor_t xDesc,
                                const void *x, const void *beta, const ccTensorDescriptor_t dxDesc, void *dx);
#endif

ccStatus_t ccL2LossForward(ccHandle_t handle, const ccL2LossDescriptor_t l2lossDesc, const void *alpha,
                           const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                           const ccTensorDescriptor_t yDesc, void *y);

/**
 * @ingroup dnn
 * @brief get the output dimension info of top k v2
 * @param [in] xDesc            descriptor of input tensor x
 * @param [in] yDesc            descriptor of input tensor y
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetTopKV2OutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t kDesc, const void *k,
                                const int64_t axis, int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief top k v2 forward computation
 * @param [in] handle           cce handle
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor x
 * @param [in] x                input data x in device memory
 * @param [in] yDesc            descriptor of input tensor y
 * @param [in] y                input data y in device memory
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccTopKV2Forward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                           const ccTensorDescriptor_t kDesc, const void *k, const void *beta, const bool sorted,
                           const int64_t axis, void *workSpace, const uint32_t workSpaceSizeInBytes,
                           const ccTensorDescriptor_t outputValuesDesc, void *outputValues,
                           const ccTensorDescriptor_t outputIndicesDesc, void *outputIndices);

/**
 * @ingroup dnn
 * @brief get the workspace size of top k v2
 * @param [in] xDesc            descriptor of input tensor x
 * @param [in] yDesc            descriptor of input tensor y
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] sizeInBytes  point to workspace size
 * @return ccStatus_t
 */
ccStatus_t ccGetTopKV2ForwardWorkspaceSize(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t kDesc,
                                           const ccTensorDescriptor_t indiceDesc, const void *k, const int64_t axis,
                                           uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief Get unsorted segment reduction output dim
 * @param [in] xDesc                 descriptor of input tensor
 * @param [in] segmentIdsDesc        descriptor of input segmentIds tensor
 * @param [in] segmentsNum           output slice num
 * @param [out] dimCnt               output tensor dim cnt
 * @param [out] dim                  output tensor dim
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetUnsortedSegmentReductionOutputDim(const ccTensorDescriptor_t xDesc,
                                                  const ccTensorDescriptor_t segmentIdsDesc, int32_t segmentsNum,
                                                  int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief reduce all forward computation
 * @param [in] handle          cce handle
 * @param [in] segmentsNum     output slice num
 * @param [in] alpha           scaling factors
 * @param [in] xDesc           descriptor of input tensor
 * @param [in] x               input data in device memory
 * @param [in] segmentIdsDesc  descriptor of input segmentIds tensor
 * @param [in] x               input segmentIds data in device memory
 * @param [in] beta            bias factors
 * @param [in] outputDesc      descriptor of output tensor
 * @param [in|out] output      output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccUnsortedSegmentSumForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc,
                                       const void *x, const ccTensorDescriptor_t segmentIdsDesc, const void *segmentIds,
                                       const int32_t segmentsNum, const void *beta,
                                       const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief reverse sequence forward computation
 * @param [in] handle           cce handle
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor x
 * @param [in] x                input data x in device memory
 * @param [in] yDesc            descriptor of input tensor y
 * @param [in] y                input data y in device memory
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccReverseSequenceForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t inputDesc,
                                    const void *input, const ccTensorDescriptor_t seqLengthsDesc,
                                    const void *seqLengths, int64_t seqAxis, int64_t batchAxis, const void *beta,
                                    const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief realdiv between two tensors.
 * @param [in] alpha  reserved.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] x  data point of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [in] y  data point of the right operator tensor.
 * @param [in] beta  reserved.
 * @param [in] outputDesc  description of the output tensor.
 * @param [output] output  data point of the output tensor.
 * @return ccStatus_t
 */

ccStatus_t ccEqualForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                          const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get output shape of realdiv.
 * @param [in] xDesc  description of the left operator tensor.
 * @param [in] yDesc  description of the right operator tensor.
 * @param [out] dimCnt       output tensor dim cnt
 * @param [out] dim          output tensor dim
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetEqualOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                               int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief invert permutation forward computation
 * @param [in] handle           cce handle
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] x                input data in device memory
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccInvertPermutationForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc,
                                      const void *x, const void *beta, const ccTensorDescriptor_t outputDesc,
                                      void *output);

/**
 * @ingroup dnn
 * @brief get the workspace size of non max suppression
 * @param [in] handle            descriptor of handle
 * @param [in] scoresDesc        descriptor of input tensor scoresDesc
 * @param [in] boxesDesc         descriptor of input tensor boxesDesc
 * @param [in|out] sizeInBytes   point to workspace size
 * @return ccStatus_t
 */
ccStatus_t ccGetNonMaxSuppressionWorkspaceSize(ccHandle_t handle, const ccTensorDescriptor_t scoresDesc,
                                               const ccTensorDescriptor_t boxesDesc, uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief get the output dim of non max suppression
 * @param [in] scoresDesc            descriptor of input tensor scoresDesc
 * @param [in] maxOutPutSize         the max size of output
 * @param [in|out] dimCnt            point to the count of dim
 * @param [in|out] dim[]             the array of output dim
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetNonMaxSuppressionOutputDim(const ccTensorDescriptor_t scoresDesc, const int32_t maxOutPutSize,
                                           int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief multinomial forward.
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x        input data in device memory
 * @param [in] numSamples    number of independent samples to draw for each row slice
 * @param [in] seed1   sed to create a random seed for the distribution
 * @param [in] seed2  sed to create a random seed for the distribution
 * @param [in] workSpace  work space for inter access
 * @param [in] workSpaceSizeInBytes  work space size
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccMultinomialForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                int32_t numSamples, int64_t seed1, int64_t seed2, void *workSpace,
                                uint32_t workSpaceSizeInBytes, const void *beta, const ccTensorDescriptor_t outputDesc,
                                void *output);
/**
 * @ingroup dnn
 * @brief get output dim of generated one-hot tensor.
 * @param [in] indicesDesc   Indices description of one-hot position.
 * @param [in] depth         On/off value description.
 * @param [in] axis          Data pointer of on/off value.
 * @param [output] dimCnt    Description of the generated one-hot tensor.
 * @param [output] dim       Data pointer of output.
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetOneHotOutputDim(const ccTensorDescriptor_t indicesDesc, int32_t depth, int32_t axis, int32_t *dimCnt,
                                int32_t *dim, int32_t dimLen);

/**
 * @ingroup dnn
 * @brief generate an one-hot Tensor use given on/off value.
 * @param [in] handle        Stream handle.
 * @param [in] alpha         reserved.
 * @param [in] indicesDesc   Indices description of one-hot position.
 * @param [in] indices       Data pointer of indices.
 * @param [in] onDesc        On value description.
 * @param [in] on            Data pointer of on value.
 * @param [in] offDesc       Off value description.
 * @param [in] off           Data pointer of off value.
 * @param [in] depth         On/off value description.
 * @param [in] axis          Data pointer of on/off value.
 * @param [in] beta          reserved.
 * @param [in] outputDesc    Description of the generated one-hot tensor.
 * @param [output] output    Data pointer of output.
 * @return ccStatus_t
 */
ccStatus_t ccOneHotForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t indicesDesc,
                           const void *indices, const ccTensorDescriptor_t onDesc, const void *on,
                           const ccTensorDescriptor_t offDesc, const void *off, const int32_t depth, const int32_t axis,
                           const void *beta, const ccTensorDescriptor_t outputDesc, void *output);
/**
 * @ingroup dnn
 * @brief get the workspaceSize of multinomial
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] numSamples       number sample
 * @param [out] sizeInBytes       wor space size of byte
 * @return ccStatus_t
 */
ccStatus_t ccGetMultinomialWorkspaceSize(const ccTensorDescriptor_t xDesc, uint32_t *sizeInBytes);
/**
 * @ingroup dnn
 * @brief get the output dimension info of multinomial
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] numSample        number of independent samples to draw for each row slice
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetMultinomialOutputDim(const ccTensorDescriptor_t xDesc, int32_t numSample, int32_t *dimCnt,
                                     int32_t dim[], int32_t dimLen);
/**
 * @ingroup dnn
 * @brief get the output dimension info of BiasAddBackward
 * @param [in] dyDesc            descriptor of input tensor
 * @param [in] out] n             outputTensor [N]CHW
 * @param [in|out] c             outputTensor N[C]HW
 * @param [in|out] h             outputTensor NC[H]W
 * @param [in|out] w            outputTensor NCH[W]
 * @return ccStatus_t
 */
ccStatus_t ccGetBiasAddBackwardOutputDim(const ccTensorDescriptor_t dyDesc, int32_t *n, int32_t *c, int32_t *h,
                                         int32_t *w);

/**
 * @ingroup dnn
 * @brief biasadd backward.
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] dyDesc       descriptor of input data
 * @param [in] dy       input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] dbDesc   descriptor of output data
 * @param [in|out] db   output data in device memory
 * @return ccStatus_t
 */
#ifndef DAVINCI_LITE
ccStatus_t ccBiasAddBackward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t dyDesc, const void *dy,
                             const void *beta, const ccTensorDescriptor_t dbDesc, void *db);

ccStatus_t ccMaxPoolWithArgmaxForward(ccHandle_t handle, const ccPoolingDescriptor_t poolingDesc, const void *alpha,
                                      const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                      const ccTensorDescriptor_t yDesc, void *y, const ccTensorDescriptor_t argMaskDesc,
                                      void *argMask);
#endif

ccStatus_t ccCreatePoolingMaskDescriptor(ccTensorDescriptor_t *poolingMaskDesc);

ccStatus_t ccDestroyPoolingMaskDescriptor(ccTensorDescriptor_t *poolingMaskDesc);

ccStatus_t ccSetPoolingMaskTensorDescriptor(ccTensorDescriptor_t poolingMaskDesc, ccTensorFormat_t format,
                                            ccDataType_t dataType, int32_t n, int32_t c, int32_t h, int32_t w,
                                            int32_t windowH, int32_t windowW);

ccStatus_t ccGetPoolingMaskTensorSizeInBytes(ccTensorDescriptor_t poolingMaskDesc, uint32_t *size);

/**
 * @ingroup dnn
 * @brief get the mask output dimension info of maxpooling training forward
 * @param [in] pooling   descriptor of convolution operator
 * @param [in] xDesc   descriptor of input tensor
 * @param [in|out] n   point to batch size
 * @param [in|out] c   point to channels
 * @param [in|out] h   point to height of feature map
 * @param [in|out] w   point to width of feature map
 * @param [in|out] windowH   point to height of window
 * @param [in|out] windowW   point to width of windowW
 * @return ccStatus_t
 */
ccStatus_t ccGetPoolingMaskDim(const ccPoolingDescriptor_t poolingDesc, const ccTensorDescriptor_t xDesc, int32_t *n,
                               int32_t *c, int32_t *h, int32_t *w, int32_t *windowH, int32_t *windowW);

#ifndef DAVINCI_LITE
ccStatus_t ccSoftmaxCrossEntropyLoss(ccHandle_t handle, ccSoftmaxAlgo_t algo, ccSoftmaxMode_t mode,
                                     ccCrossEntropyMode_t ceMode, const void *alpha, const void *scale,
                                     const ccTensorDescriptor_t logitsDesc, const void *logits,
                                     const ccTensorDescriptor_t labelsDesc, const void *labels, const void *labelSmooth,
                                     const void *beta, const ccTensorDescriptor_t lossDesc, void *loss);

ccStatus_t ccSoftmaxCrossEntropyDx(ccHandle_t handle, ccSoftmaxAlgo_t algo, ccSoftmaxMode_t mode,
                                   ccCrossEntropyMode_t ceMode, const void *alpha, const void *scale,
                                   const ccTensorDescriptor_t logitsDesc, const void *logits,
                                   const ccTensorDescriptor_t labelsDesc, const void *labels, const void *labelSmooth,
                                   const void *beta, const ccTensorDescriptor_t dxDesc, void *dx);

ccStatus_t ccAvgPoolingBackward(ccHandle_t handle, const ccPoolingDescriptor_t poolingDesc, const void *alpha,
                                const ccTensorDescriptor_t dyDesc, const void *dy, const void *beta,
                                const ccTensorDescriptor_t dxDesc, const void *dx);

ccStatus_t ccTrainingAssignOp(ccHandle_t handle, const ccAssignOpMode_t assignOpDesc, const void *alpha,
                              const void *beta, const ccTensorDescriptor_t aDesc, void *a,
                              const ccTensorDescriptor_t bDesc, const void *b);

/**
 * @ingroup dnn
 * @brief momentum optimizer for variable update
 * @param [in] handle                       cce handle
 * @param [in] inputDesc                    descriptor of input tensor: gradient,accumulation,variable
 * @param [in] gradient                     gradient input
 * @param [in|out] accumulation             accumulation input and updated output
 * @param [in|out] variable                 variable input and updated output
 * @param [in] algo                         indicate whether need FP16 output
 * @param [in] momentum                     scaler to control accumulation
 * @param [in] learningRate                 scaler
 * @param [in] lossScaleReciprocal          scaler
 * @param [in] workSpace                    additional memory address
 * @param [in] workSpaceSizeInBytes         additional memory size
 * @param [out] variableUpdatedFP16Desc     descriptor of FP16 output tensor: variableUpdatedFP16
 * @param [out] variableUpdatedFP16         variableUpdatedFP16
 * @return ccStatus_t
 */
ccStatus_t ccApplyMomentum(ccHandle_t handle, const ccTensorDescriptor_t inputDesc, const void *gradient,
                           void *accumulation, void *variable, const ccMomentumAlgo_t algo, const void *momentum,
                           const void *learningRate, const void *lossScaleReciprocal, void *workSpace,
                           const uint32_t workSpaceSizeInBytes, const ccTensorDescriptor_t variableUpdatedFP16Desc,
                           void *variableUpdatedFP16);

ccStatus_t ccSsdClassifyLossTrain(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t labelDesc,
                                  const void *label, const ccTensorDescriptor_t greaterConstDesc,
                                  const void *greaterConst, const ccTensorDescriptor_t subConstDesc,
                                  const void *subConst, const ccTensorDescriptor_t sparseDesc, const void *sparse,
                                  const void *beta, const ccTensorDescriptor_t castoutDesc, const void *castout,
                                  const ccTensorDescriptor_t muloutDesc, const void *mulout);

#endif

/**
 * @ingroup dnn
 * @brief get the workspace size of applymomentum
 * @param [in] inputDesc                    descriptor of input tensor
 * @return ccStatus_t
 */
ccStatus_t ccGetApplyMomentumWorkspaceSize(const ccTensorDescriptor_t inputDesc, uint32_t *sizeInBytes);
#ifndef DAVINCI_LITE
ccStatus_t ccHwck2FracZ(ccHandle_t handle, const ccFilterDescriptor_t xDesc, const void *x,
                        const ccFilterDescriptor_t yDesc, void *y);

ccStatus_t ccFracZ2Hwck(ccHandle_t handle, const ccFilterDescriptor_t xDesc, const void *x,
                        const ccFilterDescriptor_t yDesc, void *y);
ccStatus_t ccAddNForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const int32_t inputNum,
                         const void *x[], const void *beta, void *workSpace, uint32_t workSpaceSizeInBytes,
                         const ccTensorDescriptor_t yDesc, void *y);
#endif
ccStatus_t ccGetAddNForwardWorkspaceSize(ccHandle_t handle, const ccTensorDescriptor_t xDesc, const int32_t inputNum,
                                         const ccTensorDescriptor_t yDesc, uint32_t *sizeInBytes);
ccStatus_t ccGetAddNForwardOutputDim(const ccTensorDescriptor_t xDesc, int32_t *dimCnt, int32_t *dim, int32_t dimLen);
ccStatus_t ccAddTrainForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                             const ccTensorDescriptor_t wDesc, const void *w, const void *beta, void *workSpace,
                             uint32_t workSpaceSizeInBytes, const ccTensorDescriptor_t yDesc, void *y);
ccStatus_t ccGetAddTrainForwardWorkspaceSize(ccHandle_t handle, const ccTensorDescriptor_t xDesc,
                                             const ccTensorDescriptor_t wDesc, const ccTensorDescriptor_t yDesc,
                                             uint32_t *sizeInBytes);
ccStatus_t ccGetAddTrainForwardOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t wDesc,
                                         int32_t *dimCnt, int32_t dim[], int32_t dimLen);
ccStatus_t ccMulTrainForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                             const ccTensorDescriptor_t wDesc, const void *w, const void *beta, void *workSpace,
                             uint32_t workSpaceSizeInBytes, const ccTensorDescriptor_t yDesc, void *y);
ccStatus_t ccGetMulTrainForwardWorkspaceSize(ccHandle_t handle, const ccTensorDescriptor_t xDesc,
                                             const ccTensorDescriptor_t wDesc, const ccTensorDescriptor_t yDesc,
                                             uint32_t *sizeInBytes);
ccStatus_t ccGetMulTrainForwardOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t wDesc,
                                         int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief get workspace size
 * @param [in] xDesc            descriptor of input tensor
 * @param [in|out] sizeInBytes  workspace size
 * @return ccStatus_t
 */
ccStatus_t ccGetRandomShuffleWorkspaceSize(const ccTensorDescriptor_t xDesc, uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief random shuffle forward computation
 * @param [in] handle               cce handle
 * @param [in] alpha                common scale factor
 * @param [in] xDesc                descriptor of input data
 * @param [in] x                    input data in device memory
 * @param [in] workspace            temporary space
 * @param [in] workspaceSizeInBytes temporary space size
 * @param [in] seed                 random seed used to generate random number
 * @param [in] seed2                random seed used to generate random number
 * @param [in] beta                 common scale factor
 * @param [in] outputDesc           descriptor of output data
 * @param [in|out] output           output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccRandomShuffleForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                  void *workspace, const uint32_t workspaceSizeInBytes, const int64_t seed1,
                                  const int64_t seed2, const void *beta, const ccTensorDescriptor_t outputDesc,
                                  void *output);
/**
 * @ingroup dnn
 * @brief sin forward:
 *          data type only support float float16 double
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] input        input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccSinForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *input,
                        const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief cos forward:
 *          data type only support float float16 double
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] input        input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccCosForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *input,
                        const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief tan forward:
 *          data type only support float float16 double
 *          data format only support ND
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] input        input data in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccTanForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *input,
                        const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief get the output dimension info of unstack
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] axis             the axis to unstack along
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetUnstackOutputDim(const ccTensorDescriptor_t xDesc, int32_t axis, int32_t *dimCnt, int32_t dim[],
                                 int32_t dimLen);

/**
 * @ingroup dnn
 * @brief unstack forward.
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data
 * @param [in] x            input data in device memory
 * @param [in] num          the length of the dimension axis
 * @param [in] axis         the axis to unstack along
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */

ccStatus_t ccUnstackForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                            int32_t num, int32_t axis, const void *beta, const ccTensorDescriptor_t outputDesc,
                            void *output[]);

ccStatus_t ccResizeNearestNeighborCpuForward(ccHandle_t handle, const ccResizeNearestNeighborDescriptor_t resizeDesc,
                                             const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                             const void *beta, const ccTensorDescriptor_t outputDesc, void *output);
/**
 * @ingroup dnn
 * @brief get the output dimension info of resize nearest neighbor
 * @param [in] resizeDesc       descriptor of resize
 * @param [in] xDesc            descriptor of input tensor
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetResizeNearestNeighborOutputDim(const ccResizeNearestNeighborDescriptor_t resizeDesc,
                                               const ccTensorDescriptor_t xDesc, int32_t *dimCnt, int32_t dim[],
                                               int32_t dimLen);

/**
 * @ingroup dnn
 * @brief create descriptor of ResizeNearestNeighbor
 * @param [in|out] resizeDesc   point to descriptor of ResizeNearestNeighbor attr
 * @return ccStatus_t
 */
ccStatus_t ccCreateResizeNearestNeighborDescriptor(ccResizeNearestNeighborDescriptor_t *resizeDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of ResizeNearestNeighbor
 * @param [in|out] resizeDesc   point to descriptor of ResizeNearestNeighbor attr
 * @return ccStatus_t
 */
ccStatus_t ccDestroyResizeNearestNeighborDescriptor(ccResizeNearestNeighborDescriptor_t *resizeDesc);

/**
 * @ingroup dnn
 * @brief set descriptor of ResizeNearestNeighbor.
 * @param [in|out] resizeDesc           descriptor of resize nearest neighbor operator
 * @param [in] alignCorners             whether the centers of input and output are aligned
 * @param [in] height                   height of output
 * @param [in] width                    width of output
 * @return ccStatus_t
 */
ccStatus_t ccSetResizeNearestNeighborDescriptor(ccResizeNearestNeighborDescriptor_t resizeDesc, bool alignCorners,
                                                int32_t height, int32_t width);

/**
 * @ingroup dnn
 * [ccGetPadV2OutputDim]
 * @brief get the output dimension info of pad
 * @param [in] xDesc            descriptor of input tensor x
 * @param [in] padDesc          descriptor of input paddings
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetPadV2OutputDim(const ccTensorDescriptor_t xDesc, const ccPadV2Descriptor_t padDesc, int32_t *dimCnt,
                               int32_t dim[], int32_t dimLen);

ccStatus_t ccPadV2CpuForward(ccHandle_t handle, const ccPadV2Descriptor_t padDesc, const void *alpha,
                             const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                             const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief create descriptor of parameters for padv2 function
 * @param [in] point to descriptor of parameters for padv2 function
 * @return ccStatus_t
 */
ccStatus_t ccCreatePadV2Descriptor(ccPadV2Descriptor_t *padDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of parameters for padv2 function
 * @param [in] point to descriptor of parameters for padv2 function
 * @return ccStatus_t
 */
ccStatus_t ccDestroyPadV2Descriptor(ccPadV2Descriptor_t *padDesc);

/**
 * @brief init descriptor for parameter of padv2 function
 * @param [in|out] padDesc   descriptor of pad
 * @param [in] padShapeCnt   padshape count
 * @param [in] padShapeLow   padshape low
 * @param [in] padShapeHigh  padshape high
 * @param [in] padMode       pad mode
 * @param [in] padValue      pad value ptr
 * @param [in] padValueType  pad value data type
 * @return ccStatus_t
 */
ccStatus_t ccSetPadV2Descriptor(ccPadV2Descriptor_t padDesc, const int32_t padShapeCnt, const int32_t padShapeLow[],
                                const int32_t padShapeHigh[], const ccPadMode_t padMode, const void *padValue,
                                const ccDataType_t padValueType);
/**
 * @ingroup dnn
 * @brief create descriptor of batchToSpace
 * @param [in|out] batchToSpaceDesc  point to descriptor of batchToSpace
 * @return ccStatus_t
 */
ccStatus_t ccCreateBatchToSpaceDescriptor(ccBatchToSpaceDescriptor_t *batchToSpaceDesc);

/**
 * @ingroup dnn
 * @brief set batchToSpaceDesc
 * @param [in|out] batchToSpaceDesc descriptor of batchToSpace
 * @param [in] blockShape  blockShape of batchToSpace
 * @param [in] crops  crops of batchToSpace
 * @param [in] blockShapeLength  blockShapeLength of batchToSpace
 * @return ccStatus_t
 */
ccStatus_t ccSetBatchToSpaceDescriptor(ccBatchToSpaceDescriptor_t paramsDesc, const int32_t *blockShape,
                                       const int32_t *crops, const int32_t blockShapeLength);

/**
 * @ingroup dnn
 * @brief get batchToSpaceDesc
 * @param [in|out] batchToSpaceDesc descriptor of batchToSpace
 * @param [in] blockShape  blockShape of batchToSpace
 * @param [in] crops  crops of batchToSpace
 * @param [in] blockShapeLength  blockShapeLength of batchToSpace
 * @return ccStatus_t
 */
ccStatus_t ccGetBatchToSpaceDescriptor(const ccBatchToSpaceDescriptor_t paramsDesc, int32_t *blockShape, int32_t *crops,
                                       int32_t *blockShapeLength);

/**
 * @ingroup dnn
 * @brief destroy descriptor of batchToSpace
 * @param [in] *batchToSpaceDesc descriptor of batchToSpace
 * @return ccStatus_t
 */
ccStatus_t ccDestroyBatchToSpaceDescriptor(ccBatchToSpaceDescriptor_t *batchToSpaceDesc);

/**
 * @ingroup dnn
 * @brief get the output dimension info of batch to space
 * @param [in] xDesc            descriptor of input tensor
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */

ccStatus_t ccGetBatchToSpaceOutputDim(const ccTensorDescriptor_t xDesc,
                                      const ccBatchToSpaceDescriptor_t batchToSpaceDesc, int32_t *dimCnt, int32_t dim[],
                                      int32_t dimLen);

/**
 * @ingroup dnn
 * @brief batch to space forward computation
 * @param [in] handle           cce handle
 * @param [in] paramsDesc       descriptor of input params
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] x                input data in device memory
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */

ccStatus_t ccBatchToSpaceForward(ccHandle_t handle, const ccBatchToSpaceDescriptor_t paramsDesc, const void *alpha,
                                 const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                 const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief create descriptor of spaceToBatch
 * @param [in|out] spaceToBatchDesc  point to descriptor of spaceToBatch
 * @return ccStatus_t
 */
ccStatus_t ccCreateSpaceToBatchDescriptor(ccSpaceToBatchDescriptor_t *spaceToBatchDesc);

/**
 * @ingroup dnn
 * @brief set spaceToBatchDesc
 * @param [in|out] spaceToBatchDesc descriptor of spaceToBatch
 * @param [in] blockShape  blockShape of spaceToBatch
 * @param [in] paddings  paddings of spaceToBatch
 * @param [in] blockShapeLength  blockShapeLength of spaceToBatch
 * @return ccStatus_t
 */
ccStatus_t ccSetSpaceToBatchDescriptor(ccSpaceToBatchDescriptor_t paramsDesc, const int32_t *blockShape,
                                       const int32_t *paddings, const int32_t blockShapeLength);

/**
 * @ingroup dnn
 * @brief get spaceToBatchDesc
 * @param [in|out] spaceToBatchDesc descriptor of spaceToBatch
 * @param [in] blockShape  blockShape of spaceToBatch
 * @param [in] paddings  paddings of spaceToBatch
 * @param [in] blockShapeLength  blockShapeLength of spaceToBatch
 * @return ccStatus_t
 */
ccStatus_t ccGetSpaceToBatchDescriptor(const ccSpaceToBatchDescriptor_t paramsDesc, int32_t *blockShape,
                                       int32_t *paddings, int32_t *blockShapeLength);

/**
 * @ingroup dnn
 * @brief destroy descriptor of spaceToBatch
 * @param [in] *spaceToBatchDesc descriptor of spaceToBatch
 * @return ccStatus_t
 */
ccStatus_t ccDestroySpaceToBatchDescriptor(ccSpaceToBatchDescriptor_t *spaceToBatchDesc);

/**
 * @ingroup dnn
 * @brief get the output dimension info of space to batch
 * @param [in] xDesc            descriptor of input tensor
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */

ccStatus_t ccGetSpaceToBatchOutputDim(const ccTensorDescriptor_t xDesc,
                                      const ccSpaceToBatchDescriptor_t spaceToBatchDesc, int32_t *dimCnt, int32_t dim[],
                                      int32_t dimLen);

/**
 * @ingroup dnn
 * @brief space to batch forward computation
 * @param [in] handle           cce handle
 * @param [in] paramsDesc       descriptor of input params
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] x                input data in device memory
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */

ccStatus_t ccSpaceToBatchForward(ccHandle_t handle, const ccSpaceToBatchDescriptor_t paramsDesc, const void *alpha,
                                 const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                                 const ccTensorDescriptor_t outputDesc, void *output);

ccStatus_t ccTransFilterDesc2TensorDesc(ccFilterDescriptor_t wDesc, ccTensorDescriptor_t tensorDesc);

/*
 * @brief get the output dimension info of extractImagePatches
 * @param [in] xDesc            descriptor of input tensor x
 * @param [in] ksizes           ksizes array
 * @param [in] strides          strides array
 * @param [in] rates            rates array
 * @param [in] padding          padding type
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @return ccStatus_t
 */
ccStatus_t ccGetExtractImagePatchesOutputDim(const ccTensorDescriptor_t xDesc, const ccIntArray_t *ksizes,
                                             const ccIntArray_t *strides, const ccIntArray_t *rates,
                                             const ccExtractImagePatchesPadType_t padding, int32_t *dimCnt,
                                             int32_t dim[], const int32_t dimLen);

/**
 * @ingroup dnn
 * @brief cum forward.
 * @param [in] handle       cce handle
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data, dimCnt:1~8
 * @param [in] x            input data in device memory
 * @param [in] axisDesc      scale factor, dimCnt:0
 * @param [in] axis            which axis to cum calc, device memory
 * @param [in] beta         common scale factor
 * @param [in] opType         calc type, eg. sum, prod....
 * @param [in] exclusive       cum  flag, true or false
 * @param [in] reverse         cum  flag, true or false
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccCumForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                        const ccTensorDescriptor_t axisDesc, const void *axis, const void *beta, const CumOpType opType,
                        const bool exclusive, const bool reverse, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @ingroup dnn
 * @brief ExtractImagePatches forward.
 * @param [in] handle       cce handle
 * @param [in] ksizes       ksizes array
 * @param [in] strides      strides array
 * @param [in] rates        rates array
 * @param [in] padding      padding type
 * @param [in] alpha        common scale factor
 * @param [in] xDesc        descriptor of input data x
 * @param [in] x            input data x in device memory
 * @param [in] beta         common scale factor
 * @param [in] outputDesc   descriptor of output data
 * @param [in|out] output   output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccExtractImagePatchesForward(ccHandle_t handle, const ccIntArray_t *ksizes, const ccIntArray_t *strides,
                                        const ccIntArray_t *rates, const ccExtractImagePatchesPadType_t padding,
                                        const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                                        const void *beta, const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @brief get argmax output dim info
 * @param [in] argDesc          argmaxmin descriptor
 * @param [in] xDesc            descriptor of input tensor
 * @param [in|out] dimCnt       output dim count
 * @param [in|out] dim          output dim
 * @param [in| dimlen        length of dim
 * @return ccStatus_t
 */
ccStatus_t ccGetArgMaxOutputDim(const ccArgmaxminDescriptor_t argDesc, const ccTensorDescriptor_t xDesc,
                                int32_t *dimCnt, int32_t dim[], int32_t dimLen);

/**
 * @ingroup dnn
 * @brief argmax forward computation
 * @param [in] handle           cce handle
 * @param [in] argDesc          argmaxmin descriptor
 * @param [in] alpha            scaling factors
 * @param [in] xDesc            descriptor of input tensor
 * @param [in] x                input data in device memory
 * @param [in] workSpace        workspace pointer
 * @param [in] workSpaceSizeInBytes   workspace size in bytes
 * @param [in] beta             bias factors
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccArgMaxForward(ccHandle_t handle, const ccArgmaxminDescriptor_t argDesc, const void *alpha,
                           const ccTensorDescriptor_t xDesc, const void *x, void *workSpace,
                           const uint32_t workSpaceSizeInBytes, const void *beta, const ccTensorDescriptor_t outputDesc,
                           void *output);

/**
 * @ingroup dnn
 * @brief get the output dimension info of argmaxmin
 * @param [in] argDesc          descriptor of tagCcArgmaxmin
 * @param [in] xDesc            descriptor of input tensor
 * @param [in|out] sizeInBytes  workspace size
 * @return ccStatus_t
 */
ccStatus_t ccGetArgMaxWorkspaceSize(const ccArgmaxminDescriptor_t argDesc, const ccTensorDescriptor_t xDesc,
                                    uint32_t *sizeInBytes);

/**
 * @ingroup dnn
 * @brief create descriptor of Argmaxmin
 * @param [in|out] resizeDesc   point to descriptor of Argmaxmin attr
 * @return ccStatus_t
 */
ccStatus_t ccCreateArgmaxminDescriptor(ccArgmaxminDescriptor_t *argDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of Interp
 * @param [in|out] resizeDesc   point to descriptor of Argmaxmin attr
 * @return ccStatus_t
 */
ccStatus_t ccDestroyArgmaxminDescriptor(ccArgmaxminDescriptor_t *argDesc);

/**
 * @ingroup dnn
 * @brief destroy descriptor of Interp
 * @param [in|out] argDesc      descriptor of tagCcArgmaxmin
 * @param [in] axisType
 * @param [in] outMaxVal        whether to return the maximum value
 * @param [in] topK             number that returns the maximum index or maximum value
 * @param [in] axis             Describes which axis of the input Tensor to reduce across
 * @param [in] keepDims         whether to keep reduced dim
 * @param [in] reduceSize       the num of elements to be reduce to get topK elements, reduceSize=-1 means the total num
 * of elements in axis dimension
 * @param [in] reduceStride     the stride for reduce operation, reduceStride=1 means the layout of target data is
 * continuous
 * @return ccStatus_t
 */
ccStatus_t ccSetArgmaxminDescriptor(ccArgmaxminDescriptor_t argDesc, int32_t axisType, bool outMaxVal, int64_t topK,
                                    int64_t axis, bool keepDims, int64_t reduceSize = -1, int64_t reduceDStride = 1);

ccStatus_t ccArgMinForward(ccHandle_t handle, const ccArgmaxminDescriptor_t argDesc, const void *alpha,
                           const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                           const ccTensorDescriptor_t outputDesc, void *output);

ccStatus_t ccGetArgMinOutputDim(const ccArgmaxminDescriptor_t argDesc, const ccTensorDescriptor_t xDesc,
                                int32_t *dimCnt, int32_t dim[], const int32_t dimLen);
/**
 * @ingroup dnn
 * @brief lsh projection forward computation
 * @param [in] handle           cce handle
 * @param [in] alpha            scaling factors
 * @param [in] hashDesc         descriptor of input tensor hashDesc
 * @param [in] hash             input data hash in device memory
 * @param [in] weightDesc       descriptor of input tensor weightDesc
 * @param [in] weight           input data weight in device memory
 * @param [in] inputDesc       descriptor of input tensor inputDesc
 * @param [in] lookup           input data lookup in device memory
 * @param [in] type             1:SPARSE 2.DENSE
 * @param [in] beta             bias factors
 * @param [in] workSpace          workSpace data in device memory
 * @param [in] workSpaceSizeInBytes   workSpace length
 * @param [in] outputDesc       descriptor of output tensor
 * @param [in|out] output       output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccLshProjectionForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t hashDesc,
                                  const void *hash, const ccTensorDescriptor_t weightDesc, const void *weight,
                                  const ccTensorDescriptor_t inputDesc, const void *input, const LSHProjectionType type,
                                  const void *beta, void *workSpace, const uint32_t workSpaceSizeInBytes,
                                  const ccTensorDescriptor_t outputDesc, void *output);
/**
 * @ingroup dnn
 * @brief get the workspace size of lsh projection
 * @param [in] inputDesc         descriptor of input tensor input
 * @param [in] hashDataType      data type of hash
 * @param [in|out] sizeInBytes   workspace size
 * @return ccStatus_t
 */
ccStatus_t ccGetLshProjectionForwardWorkspaceSize(const ccTensorDescriptor_t inputDesc, const ccDataType_t hashDataType,
                                                  uint32_t *sizeInBytes);
/**
 * @ingroup dnn
 * @brief get the output dimension info of LshProjection,
 * @param [in] hashDesc         descriptor of hash
 * @param [in] type             type of mode
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in] dimLen           dim length
 * @return ccStatus_t
 */
ccStatus_t ccGetLshProjectionOutputDim(const ccTensorDescriptor_t hashDesc, const LSHProjectionType type,
                                       int32_t *dimCnt, int32_t dim[], const int32_t dimLen);
/**
 * @ingroup dnn
 * @brief get the weight dimension info of LshProjection,
 * @param [in] inputDesc          descriptor of input
 * @param [in|out] dimCnt       point to the weight dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in] dimLen           dim length
 * @return ccStatus_t
 */
ccStatus_t ccGetLshProjectionWeightDim(const ccTensorDescriptor_t inputDesc, int32_t *dimCnt, int32_t dim[],
                                       const int32_t dimLen);

/**
 * @ingroup dnn
 * @brief init descriptor for parameter of upsample function
 * @param [in] handle                 cce handle
 * @param [in] upsamplePara           input para in host memory
 * @param [in] alpha                  common scale factor
 * @param [in] bottomDesc             descriptor of input data bottomDesc
 * @param [in] bottom                 input data bottom in device memory
 * @param [in] bottomMaskDesc         descriptor of input data bottomMaskDesc
 * @param [in] bottomMask             input data bottomMask in device memory
 * @param [in] beta                   common scale factor
 * @param [in] outputDesc             descriptor of output data
 * @param [in|out] output             output data in device memory
 * @return ccStatus_t
 */
ccStatus_t ccUpsampleForward(ccHandle_t handle, const ccUpsampleParaDescriptor_t upsamplePara, const void *alpha,
                             const ccTensorDescriptor_t bottomDesc, const void *bottom,
                             const ccTensorDescriptor_t bottomMaskDesc, const void *bottomMask, const void *beta,
                             const ccTensorDescriptor_t outputDesc, void *output);

/**
 * @brief creat descriptor for parameter of usample function
 * @param [in|out] upsampleDesc   descriptor of upsamplepara
 * @return ccStatus_t
 */
ccStatus_t ccCreateUpsampleDescriptor(ccUpsampleParaDescriptor_t *upsampleDesc);

/**
 * @brief destroy descriptor for parameter of upsample function
 * @param [in|out] upsampleDesc   descriptor of upsamplepara
 * @return ccStatus_t
 */
ccStatus_t ccDestroyUpsampleDescriptor(ccUpsampleParaDescriptor_t *upsampleDesc);

/**
 * @brief set descriptor for parameter of upsample function
 * @param [in|out] upsampleDesc   descriptor of upsamplepara
 * @param [in] scale              the scale of height and width
 * @param [in] scaleHeight        the scale of height
 * @param [in] scaleWidth         the scale of Width
 * @param [in] upsampleHeight     the height of output
 * @param [in] upsampleWidth      the width of output
 * @param [in] padOutHeight       pad value height
 * @param [in] padOutWidth        pad value width
 * @return ccStatus_t
 */
ccStatus_t ccSetUpsampleDescriptor(ccUpsampleParaDescriptor_t upsampleDesc, const int32_t scale,
                                   const int32_t scaleHeight, const int32_t scaleWidth, const int32_t upsampleHeight,
                                   const int32_t upsampleWidth, const bool padOutHeight, const bool padOutWidth);
/**
 * @ingroup dnn
 * @brief get the output dimension info of upsample
 * @param [in] upsamplePara     para of upsample
 * @param [in] bottomDesc       descriptor of input bottom tensor
 * @param [in|out] dimCnt       point to the output dimCnt
 * @param [in|out] dim          arrays to save dims
 * @param [in] dimLen           the len of dim array
 * @return ccStatus_t
 */
ccStatus_t ccGetUpsampleOutputDim(const ccUpsampleParaDescriptor_t upsamplePara, const ccTensorDescriptor_t bottomDesc,
                                  int32_t *dimCnt, int32_t dim[], const int32_t dimLen);

#ifndef DAVINCI_LITE
ccStatus_t ccMatmul(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                    const ccTensorDescriptor_t wDesc, const void *w, const ccTensorDescriptor_t biasDesc,
                    const void *bias, const ccFullConnectFwdAlgo_t algo, void *workSpace,
                    const uint32_t workSpaceSizeInBytes, const void *beta, const ccTensorDescriptor_t yDesc, void *y,
                    const bool transposeA, const bool transposeB);
ccStatus_t ccGetMatmulOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t wDesc, int32_t *n,
                                int32_t *c, int32_t *h, int32_t *w, bool transposeA, bool transposeB);
ccStatus_t ccGetMatmulWorkspaceSize(ccHandle_t handle, const ccFullConnectFwdAlgo_t algo,
                                    const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t wDesc,
                                    const ccTensorDescriptor_t yDesc, uint32_t *sizeInBytes, bool transposeA,
                                    bool transposeB);
#endif

/**
 * @ingroup dnn
 * @brief gather_v2 function
 * @param [in] handle                cce handle
 * @param [in] alpha                 common scale factor
 * @param [in] paramsDesc            descriptor
 * @param [in] params                device memory
 * @param [in] indicesDesc           descriptor
 * @param [in] indices               device memory
 * @param [in] axisDesc              descriptor
 * @param [in] axis                  device memory
 * @param [in] beta                  common scale factor
 * @param [in] outputDesc            descriptor
 * @param [in|out] output            device memory
 * @return ccStatus_t
 */
ccStatus_t ccGatherV2(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t paramsDesc, const void *params,
                      const ccTensorDescriptor_t indicesDesc, const void *indices, const ccTensorDescriptor_t axisDesc,
                      const void *axis, const void *beta, const ccTensorDescriptor_t outputDesc, const void *output);

/**
 * @ingroup dnn
 * @brief memory_clear function
 * @param [in] handle                 cce handle
 * @param [in] addrSpaceSizeInBytes   addr space size
 * @param [in|out] addr               device memory
 * @return ccStatus_t
 */
ccStatus_t ccMemoryClear(ccHandle_t handle, const uint64_t addrSpaceSizeInBytes, const void *addr);

/**
 * @ingroup dnn
 * @brief check input is overflow
 * @param [in] handle      cce handle
 * @param [in] alpha       scaling factors
 * @param [in] xDesc       descriptor of input tensor
 * @param [in] x           input data in device memory
 * @param [in] yDesc       descriptor of output tensor
 * @param [in|out] y       output data in device memory
 * @param [in] beta        scaling factors
 * @return ccStatus_t
 */
ccStatus_t ccIsFinite(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                      const ccTensorDescriptor_t yDesc, const void *y, const void *beta);
};  // namespace cce

#endif  // DNN_OP_H__

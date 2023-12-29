/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 * Description: HCOM API
 * Author: ligang
 * Create: 2019-01-16
 */

#ifndef HCOM_H
#define HCOM_H

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <functional>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Get the rank number in the group.
 *
 * @param group A string identifying the group name.
 * @param rankSize A pointer identifying the rank number.
 * @return HcclResult
 */
HcclResult HcomGetRankSize(const char *group, u32 *rankSize);

/**
 * @brief Get the rank number of this rank's server within the group.
 *
 * @param group A string identifying the group name.
 * @param localRankSize A pointer identifying the rank number.
 * @return HcclResult
 */
HcclResult HcomGetLocalRankSize(const char *group, u32 *localRankSize);

/**
 * @brief Get the rank id of this rank.
 *
 * @param group A string identifying the group name.
 * @param rankId A pointer identifying the rank id.
 * @return HcclResult
 */
HcclResult HcomGetRankId(const char *group, u32 *rankId);

/**
 * @brief Get the local rank id of this rank's server within the group.
 *
 * @param group A string identifying the group name.
 * @param localRankId A pointer identifying the local rank id.
 * @return HcclResult
 */
HcclResult HcomGetLocalRankId(const char *group, u32 *localRankId);

/**
 * @brief Get the world rank id according to the group rank id.
 *
 * @param group A string identifying the group name.
 * @param groupRank An integer(u32) identifying the group rank id.
 * @param worldRank A pointer identifying the world rank id.
 * @return HcclResult
 */
HcclResult HcomGetWorldRankFromGroupRank(const char *group, u32 groupRank, u32 *worldRank);

/**
 * @brief Get the group rank id according to the world rank id.
 *
 * @param worldRank An integer(u32) identifying the world rank id.
 * @param group A string identifying the group name.
 * @param groupRank A pointer identifying the group rank id.
 * @return HcclResult
 */
HcclResult HcomGetGroupRankFromWorldRank(u32 worldRank, const char *group, u32 *groupRank);

/**
 * @brief Create group.
 *
 * @param group A string identifying the group name.
 * @param rankNum An integer(u32) identifying the number of ranks in the group.
 * @param rankIds A list identifying the ranks in the group.
 * @return HcclResult
 */
HcclResult HcomCreateGroup(const char *group, u32 rankNum, u32 *rankIds);

/**
 * @brief Destroy group
 *
 * @param group A string identifying the group name.
 * @return HcclResult
 */
HcclResult HcomDestroyGroup(const char *group);

/**
 * @brief Set the gradient split strategy with in the group, according to gradient index.
 *
 * @param group A string identifying the group name.
 * @param segmentNum An integer(u32) identifying the segments number of gradients.
 * @param IdxList A list identifying the index of end gradient in each segment.
 * @return HcclResult
 */
extern HcclResult HcomSetGradFusionByIndex(const char *group, u32 segmentNum, const u32 *IdxList);

/**
 * @brief Set the gradient split strategy with in the group, according to gradient data size.
 *
 * @param group A string identifying the group name.
 * @param segmentNum An integer(u32) identifying the segments number of gradients.
 * @param sizeList A list identifying the percent of each segment.
 * @return HcclResult
 */
extern HcclResult HcomSetGradFusionBySize(const char *group, u32 segmentNum, const float *sizeList);

/**
 * @brief optimizer offload CPU-side hcom init.
 *
 * @param rankTable A string identifying the rank table.
 * @param rankId An integer(u32) identifying the number of rank id.
 * @return HcclResult
 */
extern HcclResult HcomInitByRankTable(const char *rankTable, uint32_t rankId);

/**
 * @brief optimizer offload CPU-side hcom destroy.
 *
 * @return HcclResult
 */
extern HcclResult HcomDestroy(void);

/**
 * @brief optimizer offload CPU-side establish a link.
 *
 * @param op A pointer identifying the op desc.
 * @param request A pointer identifying the link setup handle.
 * @return HcclResult
 */
extern HcclResult HcomPrepareStart(const HcomOpDesc* op, HcomRequest* request);

/**
 * @brief optimizer offload CPU-side query link status.
 *
 * @param request A pointer identifying link setup handle.
 * @param status A pointer identifying the link status.
 * @return HcclResult
 */
extern HcclResult HcomPrepareQuery(HcomRequest request, HcomStatus* status);

/**
 * @brief optimizer offload CPU-side cancel a link.
 *
 * @param request A pointer identifying link setup handle.
 * @param status A pointer identifying the link status.
 * @return HcclResult
 */
extern HcclResult HcomPrepareCancel(HcomRequest request, HcomStatus* status);

/**
 * @brief optimizer offload CPU-side hcom send.
 *
 * @param buf A pointer identifying the send data buf.
 * @param count An integer(uint64_t) identifying the send data cout.
 * @param dataType  An integer identifying the send data type.
 * @param peerRank An integer(uint32_t) identifying the send data peer rank.
 * @param tag An integer(uint32_t) identifying the tag.
 * @param group A pointer identifying the group name.
 * @param flag An integer(uint64_t) identifying the flag.
 * @return HcclResult
 */
extern HcclResult HcomSendByOS(void* buf, uint64_t count, HcclDataType dataType, uint32_t peerRank,
    uint32_t tag, const char* group, uint64_t flag);

/**
 * @brief optimizer offload CPU-side hcom recv.
 *
 * @param buf A pointer identifying the recv data buf.
 * @param count An integer(uint64_t) identifying the recv data cout.
 * @param dataType  An integer identifying the redv data type.
 * @param peerRank An integer(uint32_t) identifying the recv data peer rank.
 * @param tag An integer(uint32_t) identifying the tag.
 * @param group A pointer identifying the group name.
 * @param flag An integer(uint64_t) identifying the flag.
 * @return HcclResult
 */
extern HcclResult HcomReceiveByOS(void* buf, uint64_t count, HcclDataType dataType, uint32_t peerRank,
    uint32_t tag, const char* group, uint64_t flag);

extern HcclResult HcomGatherByOs(void* inputBuf, uint64_t inputCount, HcclDataType inputType, void* outputBuf,
    uint64_t outputCount, HcclDataType outputType, int root, const char *group, uint64_t flag);

extern HcclResult HcomBcastByOS(void* buf, uint64_t count, HcclDataType dataType, int root, const char *group,
    uint64_t flag);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCOM_H

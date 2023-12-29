/**
 * @file progress.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */
#ifndef ACG_PROGRESS_H
#define ACG_PROGRESS_H
#include <mutex>
#include <string>
#include <memory>
#include <unordered_set>
#include <functional>

namespace Acg {
constexpr float MAX_PROGRESS_WEIGHT           = 100;
constexpr float MID_PROGRESS_VALUE            = 50;
constexpr float MAX_PROGRESS_VALUE            = 100;
constexpr float DEFAULT_PROGRESS_VALUE        = 0;

enum class Status {
    SUCCESS,
    INVALID_PARAMETER,
    NOT_FOUND_MARK,
    OUT_OF_WEIGHT_RANGE,
    OUT_OF_SIZE_LIMIT,
};

using ProgressReportFunc = std::function<void(const std::string &msg, float value)>;

struct ProgressAttr {
    ProgressAttr() : weight(MAX_PROGRESS_WEIGHT), value(DEFAULT_PROGRESS_VALUE) {};
    float weight;
    float value;
};

class Progress : public std::enable_shared_from_this<Progress> {
public:
    explicit Progress(const std::string &tag)
        : selfTag_(tag), prevTag_(""), progressFunc_(nullptr) {}
    ~Progress();
    /**
     * @brief      set progress report function
     * @param [in] func     : progress report function
     * @return     success == SUCCESS; failed != SUCCESS
     */
    Status SetProgressReport(const ProgressReportFunc &func);

    /**
     * @brief      set progress attribute
     * @param [in] attr     : progress attribute
     * @return     success == SUCCESS; failed != SUCCESS
     */
    Status SetAttr(const ProgressAttr &attr);

    /**
     * @brief      associate previous progress
     * @param [in] prevTag      : previous progress mark
     * @return     success == SUCCESS; failed != SUCCESS
     */
    Status AssociateProgress(const std::string &prevTag);

    /**
     * @brief      report progress value and message
     * @param [in] msg      : progress message
     * @param [in] value    : progress value
     * @return     success == SUCCESS; failed != SUCCESS
     */
    Status ReportProgress(const std::string &msg, float value = 0.0);

    /**
     * @brief      get progress mark
     * @return     progress mark
     */
    const std::string GetSelfTag() const;

    /**
     * @brief      get progress value
     * @return     0~100
     */
    const ProgressAttr GetProgressAttr() const;

private:
    /**
     * @brief      caculate progress
     * @param [in] value    : progress value
     * @return     success is [0.00, 100.00]; other is failed
     */
    float CaculateProgress(float value);

    /**
     * @brief      add sub progress
     * @param [in] subProgress    : sub progress
     * @return     success == SUCCESS; failed != SUCCESS
     */
    Status AddSubProgress(const std::shared_ptr<Progress> &subProgress);

    std::string selfTag_;
    std::string prevTag_;
    ProgressReportFunc progressFunc_;
    ProgressAttr attr_;
    std::unordered_set<std::shared_ptr<Progress>> subProgressSet_;
    std::recursive_mutex rmtx_;
    std::mutex mtx_;
};

using ProgressPtr = std::shared_ptr<Progress>;

/**
 * @brief      get progress of tag
 * @param [in] tag      : progress mark
 * @return     success != nullptr; failed == nullptr
 */
ProgressPtr GetProgress(const std::string &tag);

/**
 * @brief      create progress by tag
 * @param [in] tag      : progress mark
 * @return     success != nullptr; failed == nullptr
 */
ProgressPtr CreateProgress(const std::string &tag);

/**
 * @brief      associate progress of tag  to previous tag
 * @param [in] tag      : progress mark
 * @param [in] prevtag  : previous progress mark
 * @return     success == SUCCESS; failed != SUCCESS
 */
Status AssociatedProgress(const std::string &tag, const std::string &prevTag);

/**
 * @brief      destroy progress of tag
 * @param [in] tag      :  progress mark
 * @return     None
 */
void DestroyProgress(const std::string &tag);

/**
 * @brief      report progress of tag
 * @param [in] tag      : progress mark
 * @param [in] msg      : progress update message
 * @param [in] value    : progress value
 * @return     success == SUCCESS; failed != SUCCESS
 */
Status ReportProgress(const std::string &tag, const std::string &msg, float value = 0.0);

/**
 * @brief      display progress on the screen
 * @param [in] msg      : progress message
 * @param [in] value    : progress value
 * @return     success == SUCCESS; failed != SUCCESS
 */
void DisplayProgressOnScreen(const std::string &msg, float value);
}
#endif

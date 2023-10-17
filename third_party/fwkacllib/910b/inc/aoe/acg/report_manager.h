/**
 * @file report_manager.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#ifndef AOE_REPORT_MANAGER_H
#define AOE_REPORT_MANAGER_H

#include <map>
#include <mutex>
#include "nlohmann/json.hpp"
#include "aoe_types.h"

namespace Aoe {
class Report;
using ReportPtr = std::shared_ptr<Report>;

class ReportManager {
public:
    ReportManager() = default;
    ~ReportManager() = default;
    /**
     * @brief       : Obtain ReportManager instance
     * @return      : ReportManager instance
     */
    static ReportManager &GetInstance()
    {
        static ReportManager instance;
        return instance;
    }

    /**
     * @brief       : create report object
     *                filePath can be empty, single report cannot be saved at this time
     * @param [in]  : tag       tag, unified use job_id
     * @param [in]  : filePath  file Path to save
     * @return      : AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    AoeStatus CreateReport(const std::string &tag, const std::string &filePath);

    /**
     * @brief       : create report object
     *                filePath can be empty, single report cannot be saved at this time
     * @param [in]  : tag       tag, unified use job_id
     * @param [in]  : filePath  file Path to save
     * @return      : AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    AoeStatus CreateReportWithBasicMsg(const std::string &tag, const std::string &filePath,
    const nlohmann::json &object, const uint32_t priority);

    /**
     * @brief       : submit report messsage
     * @param [in]  : tag       tag, unified use job_id
     * @param [in]  : object    report messsage object
     * @param [in]  : priority  report messsage priority, limit: 0-100
     * @return      : AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    AoeStatus SubmitReport(const std::string &tag, const nlohmann::json &object, const uint32_t priority);

    /**
     * @brief       : save report file
     * @param [in]  : tag       tag, unified use job_id
     * @return      : AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    AoeStatus SaveReport(const std::string &tag);

    /**
     * @brief       : Updata Report Path to the specified path
     *                filePath can be empty, report will save to current dir/aoe_result.json
     * @param [in]  : filePath   file Path to save
     * @return      : AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    void UpdateReportPath(const std::string &filePath);

    /**
     * @brief       : summarize all completed reports to the specified path
     *                filePath can be empty, report will save to current dir/aoe_result.json
     * @param [in]  : filePath   file Path to save
     * @return      : AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    AoeStatus SummaryReports();

    /**
     * @brief       : clear all completed reports
     * @return      : AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    void ClearReports();

     /**
     * @brief      update report of tag
     * @param [in] tag      : report mark
     * @param [in] msg      : report update message
     * @return     AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    AoeStatus UpdateReport(const std::string &tag, const nlohmann::json &reportJson, const uint32_t priority);

     /**
     * @brief      update report result of tag
     * @param [in] tag      : report mark
     * @param [in] msg      : report update message
     * @return     AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    AoeStatus UpdateReportResult(const std::string &tag, const std::string &tuneResult, const uint32_t priority);
     /**
     * @brief      change result of tag
     * @param [in] tag      : report mark
     * @return     AOE_SUCCESS: == 0; AOE_FAILURE: != 0
     */
    AoeStatus ChangeReportResult(const std::string &tag);

    /**
     * @brief       : Check whether new repo is added or updated.
     * @param [in]  : tag       tag, unified use job_id
     * @param [in]  : priority  report messsage priority, limit: 0-100
     * @return      : true or flase
     */
    bool IsUpdateRepo(const std::string &tag, const uint32_t priority);

private:
    std::mutex mgrMtx_;
    std::map<std::string, ReportPtr> runtimeReports_;
    std::multimap<std::string, ReportPtr> historyReports_;
    std::string resultPath_;
};

constexpr uint32_t MAX_PRIORITY = 100;              // 90-100 reserved for aoe framework
constexpr uint32_t MAX_PLUGIN_PRIORITY = 90;        // 0-90 reserved for aoe plug-in
}
#endif
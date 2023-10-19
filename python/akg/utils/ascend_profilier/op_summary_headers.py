class OpSummaryHeaders(object):
    # op_summary
    TASK_START_TIME = "Task Start Time(us)"
    AIC_TOTAL_CYCLES = "aic_total_cycles"
    AIV_TOTAL_CYCLES = "aiv_total_cycles"
    TASK_DURATION = "Task Duration(us)"
    OP_SUMMARY_SHOW_HEADERS = ["Op Name", "OP Type", "Task Type", TASK_START_TIME, TASK_DURATION,
                               "Task Wait Time(us)", "Block Dim" ,AIC_TOTAL_CYCLES, AIV_TOTAL_CYCLES]

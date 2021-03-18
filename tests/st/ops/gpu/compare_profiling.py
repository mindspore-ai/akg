def read_file(manual, auto):
    manual_prof = dict()
    auto_prof = dict()
    cur_op = ""

    def _process_line(line, is_manual):
        nonlocal cur_op
        prof = manual_prof if is_manual else auto_prof
        op_key = "Operater:"
        time_key = "gpu(0): exec="

        if op_key in line:
            cur_op = line.split(op_key)[1].replace("\n", "").strip(" ")

        if time_key in line:
            cur_time = line.split(time_key)[1].replace(
                "\n", "").replace("sec/op", "").strip(" ")
            if cur_op not in prof:
                prof[cur_op] = [cur_time]
            else:
                prof[cur_op].append(cur_time)

    with open(manual, "r") as f:
        for line in f:
            _process_line(line, True)

    with open(auto, "r") as f:
        for line in f:
            _process_line(line, False)
    compare(manual_prof, auto_prof)


def compare(manual_prof, auto_prof):
    for k, m_times in manual_prof.items():
        a_times = auto_prof.get(k)

        if not a_times:
            print("Time for {} if not found in auto schedule".format(k))
            continue
        print("operator: {}".format(k))
        for m, a in zip(m_times, a_times):
            print("manual {} vs auto {}".format(m, a))
        print("")


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compare_profiling.py manual_log_name auto_log_name")
        sys.exit()
    read_file(sys.argv[1], sys.argv[2])

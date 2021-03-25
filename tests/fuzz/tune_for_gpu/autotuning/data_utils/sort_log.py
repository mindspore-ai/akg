import sys

if __name__ == "__main__":
    from_log_file = str(sys.argv[1])
    sorted_log_file = str(sys.argv[2])
    f_in = open(from_log_file, 'r')
    f_out = open(sorted_log_file, "wt")
    d = dict()
    for line in f_in:
        config = line.split("|")
        d[str(config[1])] = float(config[2])
    sorted_dict = {k: v for k, v in sorted(
        d.items(), key=lambda item: (item[1], item[0]))}
    for k, v in sorted_dict.items():
        f_out.write("|" + str(k) + "|" + str(v) + "\n")
    f_in.close()
    f_out.close()

#!/usr/bin/env python3

import sys

if __name__ == "__main__":

    commit_msg_file = sys.argv[1]
    with open(commit_msg_file) as f:
        commit_msg = f.read()

    commit_msg_list = list(commit_msg.strip())
    commit_msg_list[0] = commit_msg_list[0].upper()
    commit_msg_output = "".join(commit_msg_list)

    with open(commit_msg_file, "w") as f:
        f.write(commit_msg_output)

import subprocess
import os
import pathlib
import numpy as np

PROF_BOARD_DIR = "./prof_boards/"
DEPTH = 2
NUM_ROWS_PER_BOARD = 4 * DEPTH # each combo of parallel search/eval at each depth
DATA_FILE_NAME = "./data.txt"



def get_board_files():
    res = os.listdir(PROF_BOARD_DIR)
    board_files = []
    for f in res:
        if (pathlib.Path(f).suffix == ".board"):
            board_files.append(f)
    return board_files

def gather_board_data(board_files):
    cols = ['Board', 'Depth', 'Par-S', 'Par-E', 'Evals', 'Total-T', 'Eval-T']
    for i in range(DEPTH):
        cols.append('S:' + str(i))
        cols.append('P:' + str(i))

    all_data = np.zeros((len(board_files) * NUM_ROWS_PER_BOARD + 1, len(cols)), dtype=object)
    all_data[0] = np.array(cols)
    for i in range(len(board_files)):
        f = PROF_BOARD_DIR + board_files[i]
        args = ["./profile", f, str(DEPTH)]
        p = subprocess.run(args, capture_output=True)
        out = p.stdout.decode("utf-8")
        board_data = []
        lines = out.split("\n")
        for line in lines[:-1]:
            data = [float(d) for d in line.split(",")]
            board_data.append([pathlib.Path(board_files[i]).stem] + data)
        start_i = NUM_ROWS_PER_BOARD * i + 1
        all_data[start_i : NUM_ROWS_PER_BOARD + start_i] = np.array(board_data)
    return all_data


def main():
    board_files = get_board_files()
    data = gather_board_data(board_files)
    np.savetxt(DATA_FILE_NAME, data, fmt="%-8s")

    profile_data = np.array(data[1:, 1:], dtype=float)

main()
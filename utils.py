import numpy as np
import pickle

def inst_to_gpt2_inst(csv_file, size=None, skip=0, n_positions=24):
    raw = np.loadtxt(csv_file, dtype=int, max_rows=size, skiprows=skip, delimiter=',')
    x = []
    t = []
    for idx_i in range(len(raw)):
        for idx_j in range(len(raw[idx_i])-n_positions-1):
            if idx_j != 0 and raw[idx_i, idx_j+n_positions-1] == 0 and raw[idx_i, idx_j+n_positions] == 0:
                break
            x.append(raw[idx_i, idx_j: idx_j+n_positions])
            t.append(raw[idx_i, idx_j+1: idx_j+n_positions+1])
    x = np.array(x, dtype=int)
    t = np.array(t, dtype=int)
    with open('train.pkl', 'wb') as file:
        pickle.dump({'x':x,'t':t}, file)


if __name__ == "__main__":
    inst_to_gpt2_inst('./eval.txt')
__all__ = ["read_ptf_results"]

import os
import numpy as np
import pandas as pd


def read_ptf_results(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # locate "RESULTS:"
    start = next(i for i,l in enumerate(lines) if l.strip().startswith("RESULTS:"))

    # header line is two lines below
    cols = lines[start + 2].split()

    # collect numeric data
    data = []
    for line in lines[start + 4:]:
        if (not line.strip()) or "MIN ROC reached" in line or "SUCCESFULL" in line or "SUCCESSFUL" in line:
            break
        # skip horizontal separator
        if set(line.strip()) == {"="}:
            continue
        data.append(line.split())

    # build DataFrame and cast to numbers
    return pd.DataFrame(data, columns=cols).apply(pd.to_numeric)

#%%

if __name__ == '__main__':
    
    directory = os.path.join('C:\\', 'Users', 'nmb48', 'Documents', 'GitHub', 'pybada', 'examples', 'nils', 'nilsbarner_V_316_All')
    df_a20n_climb = read_ptf_results(os.path.join(directory, "V_316_A20N_CLIMB_20251004_133315.PTF"))
    
    
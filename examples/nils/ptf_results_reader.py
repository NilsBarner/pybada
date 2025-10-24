__all__ = ["read_ptf_results"]

import os
import re
import numpy as np
import pandas as pd


def read_ptf_results(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # locate "RESULTS:"
    start = next(i for i,l in enumerate(lines) if l.strip().startswith("RESULTS:"))

    # header line is two lines below
    cols = re.split(r'\s{2,}', lines[start + 2].strip())

    # collect numeric data
    data = []
    for line in lines[start + 4:]:
        if (not line.strip()) or "MIN ROC reached" in line or "SUCCESFULL" in line or "SUCCESSFUL" in line:
            break
        # skip horizontal separator
        if set(line.strip()) == {"="}:
            continue
        data.append(re.split(r'\s{2,}', line.strip()))

    # print('data, cols =', data, cols)

    # build DataFrame and cast to numbers
    return pd.DataFrame(data, columns=cols).apply(pd.to_numeric)

#%%

if __name__ == '__main__':
    df_e290_climb = read_ptf_results(os.path.join(os.getcwd(), "V_316_E290_DESCENT_20251014_125311.PTF"))
    print(df_e290_climb)


# __all__ = ["read_ptf_results"]

# import os
# import numpy as np
# import pandas as pd


# def read_ptf_results(path):
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         lines = f.readlines()

#     # locate "RESULTS:"
#     start = next(
#         i for i, l in enumerate(lines) if l.strip().startswith("RESULTS:")
#     )

#     # header line is two lines below
#     cols = lines[start + 2].split()

#     # collect numeric data
#     data = []
#     for line in lines[start + 4 :]:
#         if (
#             (not line.strip())
#             or "MIN ROC reached" in line
#             or "SUCCESFULL" in line
#             or "SUCCESSFUL" in line
#         ):
#             break
#         # skip horizontal separator
#         if set(line.strip()) == {"="}:
#             continue
#         data.append(line.split())

#     # build DataFrame and cast to numbers
#     return pd.DataFrame(data, columns=cols).apply(pd.to_numeric)


# # %%

# if __name__ == "__main__":
#     directory = os.path.join(
#         "C:\\",
#         "Users",
#         "nmb48",
#         "Documents",
#         "GitHub",
#         "pybada",
#         "examples",
#         "nils",
#         "nilsbarner_V_316_All",
#     )
#     df_a20n_climb = read_ptf_results(
#         os.path.join(directory, "V_316_A20N_CLIMB_20251004_133315.PTF")
#     )

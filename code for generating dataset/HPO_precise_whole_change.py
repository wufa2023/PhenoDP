import json
with open("../source_data/omim_to_hpo.json", "r") as json_file:
    data = json.load(json_file)
# 读取OMIM_to_HPO文件
omim_list = list(data.keys())

# 读取特异性
import pandas as pd
df = pd.read_csv('../source_data/hpo_specific.csv')
specific_list = df['hpo'].values

# 读取HPO_to_HPO关系文件
with open("../source_data/hpo_parents.json", "r") as json_file:
    omim_parents = json.load(json_file)

import random
import numpy as np
from utils import *
save_omim = []
save_hpo = []
unique_omim = []
unique_hpo = []
for _ in range(10):
    print('curr', _)
    for i in range(len(omim_list)):
        print(i)
        ref = data[omim_list[i]]
        temp = list(ref)
        new_ref = get_change(temp, specific_list, omim_parents)
        if new_ref == -1:
            continue
        else:
            save_omim.append(omim_list[i])
            save_hpo.append(new_ref)

# 去除重复
print('generate records', len(save_omim))
unique_omim, unique_hpo = remove_duplicate(save_omim_id=save_omim, save_sim_hpo=save_hpo)
print('unique records', len(unique_omim))
pd.DataFrame({'id': unique_omim, 'sim_hpo': unique_hpo}).to_csv("../source_data/precise_whole_change.csv", index=False)



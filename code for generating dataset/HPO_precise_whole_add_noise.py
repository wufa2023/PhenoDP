# used for dataset: whole precise + noise
from utils import *

# 读取precise terms
# 提取关系链条, 添加noise,
## 先找出当前OMIM HPO的前俩代的父母节点，组成一个list
## 随机从HPO表中选出特定数量的HPO，找出俩代父母节点，随后进行比对
## 挑选出的HPO俩代节点不与OMIM挑出的俩代节点重合，即认为是不相关

# read data
import json
import pandas as pd
with open('../source_data/omim_to_hpo.json') as f:
    data = json.load(f)

with open('../source_data/hpo_counts.json') as f:
    hpo_list = json.load(f)
    hpo_list = list(hpo_list.keys())

save_omim = []
save_hpo = []
unique_omim = []
unique_hpo = []
omim_id = list(data.keys())

import numpy as np
import random

for _ in range(5):
    print('iter', _)
    for i in range(len(omim_id)):
        print(i)
        ref = data[omim_id[i]]
        temp = list(ref)
        # select noise, (0.1, 0.49) random add
        choice = range(10, 50, 1)
        add_len = (random.choice(choice) / 100) * len(ref)
        _noise_len = np.round(add_len, 0)
        noise_list = []

        print('raw len', len(ref), 'add_len', _noise_len)

        while len(noise_list) < _noise_len:
            term = random.choice(hpo_list)
            if term not in ref and is_connected(ref, term) == False:
                noise_list.append(term)
            else:
                continue
        print('noise_list', noise_list)
        print('_________________________')
        temp.extend(noise_list)
        save_omim.append(omim_id[i])
        save_hpo.append(temp)
 # remove duplicate

# 去除重复
print('generate records', len(save_omim))
unique_omim, unique_hpo = remove_duplicate(save_omim_id=save_omim, save_sim_hpo=save_hpo)
print('unique records', len(unique_omim))
pd.DataFrame({'id': unique_omim, 'sim_hpo': unique_hpo}).to_csv("../source_data/precise_whole_add_noise.csv", index=False)

# save
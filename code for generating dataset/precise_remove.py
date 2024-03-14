# 创建(precise + random remove) dataset

## read omim hpo
import json
from utils import *

with open('../source_data/omim_to_hpo.json') as f:
    data = json.load(f)

import pandas as pd
df = pd.read_csv('../source_data/hpo_specific.csv')
specific_list = df['hpo'].values

save_omim_id = []
save_sim_hpo = []
omim_id = list(data.keys())

# 重复二十次
print(len(data.keys()))
for _ in range(10):
    print('iter', _)
    for i in range(len(omim_id)):
        ref = data[omim_id[i]]
        temp = list(ref)
        ## 查询omim疾病是否有特异词，至少有2个非特异词的疾病才被允许模拟生成
        ## 且omim本身总体有3个以上的词
        is_sim = is_simulation(ref_hpo_list=ref, specific_hpo=specific_list)
        if is_sim:
            ## 随机挑选10%～49%的非特异词进行删除
            sim_hpo = random_remove(ref_hpo_list=temp, specific_hpo=specific_list)
            save_omim_id.append(omim_id[i])
            save_sim_hpo.append(sim_hpo)

# 去除重复
print('generate records', len(save_omim_id))
unique_omim, unique_hpo = remove_duplicate(save_omim_id=save_omim_id, save_sim_hpo=save_sim_hpo)
print('unique records', len(unique_omim))
pd.DataFrame({'id':unique_omim, 'sim_hpo':unique_hpo}).to_csv("../source_data/precise_remove.csv", index=False)

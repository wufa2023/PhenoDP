# 创建(precise + random remove) dataset

## read omim hpo
import json
from utils import *

import pandas as pd
with open('../source_data/omim_to_hpo.json') as f:
    data = json.load(f)

df = pd.read_csv('../source_data/hpo_specific.csv')
specific_list = df['hpo'].values

df_remove = pd.read_csv('../source_data/precise_remove.csv')

with open("../source_data/hpo_parents.json", "r") as json_file:
    omim_parents = json.load(json_file)

from utils import *
save_omim = []
save_hpo = []
unique_omim = []
unique_hpo = []
omim_id = list(data.keys())

for _ in range(10):
    print('iter', _)
    for i in range(len(omim_id)):
        print(i)
        ref = data[omim_id[i]]
        temp = list(ref)
        ## 查询omim疾病是否有特异词，至少有2个非特异词的疾病才被允许模拟生成
        ## 且omim本身总体有3个以上的词
        is_sim = is_simulation(ref_hpo_list=ref, specific_hpo=specific_list)
        if is_sim:
            ## 随机挑选10%～49%的非特异词进行删除
            sim_hpo = random_remove(ref_hpo_list=temp, specific_hpo=specific_list)
            new_ref = get_change(sim_hpo, specific_list, omim_parents)
            if new_ref == -1:
                continue
            else:
                save_omim.append(omim_id[i])
                save_hpo.append(new_ref)

# 去除重复
print('generate records', len(save_omim))
unique_omim, unique_hpo = remove_duplicate(save_omim_id=save_omim, save_sim_hpo=save_hpo)
print('unique records', len(unique_omim))
pd.DataFrame({'id':unique_omim, 'sim_hpo':unique_hpo}).to_csv("../source_data/precise_remove_changed.csv", index=False)

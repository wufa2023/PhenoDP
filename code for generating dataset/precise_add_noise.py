# 读取precise terms
# 提取关系链条, 添加noise,
## 先找出当前OMIM HPO的前俩代的父母节点，组成一个list
## 随机从HPO表中选出特定数量的HPO，找出俩代父母节点，随后进行比对
## 挑选出的HPO俩代节点不与OMIM挑出的俩代节点重合，即认为是不相关

import json

with open("../source_data/omim_disease_names.json", "r") as json_file:
    omim_to_hpo = json.load(json_file)
# 读取OMIM_to_HPO文件
omim_list = list(omim_to_hpo.keys())

# 读取HPO_to_HPO关系文件
with open("../source_data/hpo_parents.json", "r") as json_file:
    omim_to_hpo = json.load(json_file)
print(len(list(omim_to_hpo.keys())))



# 读取precise terms
# 提取关系链条, 添加noise,
## 先找出当前OMIM HPO的全部三代的父母节点，组成一个list
## 随机从HPO表中选出特定数量的HPO，找出三代父母节点，随后进行比对
## 挑选出的HPO三代节点不与OMIM挑出的三代节点重合，即认为是不相关

import json

# 读取OMIM_to_HPO文件
with open("../source_data/omim_to_hpo.json", "r") as json_file:
    omim_to_hpo = json.load(json_file)
omim_list = list(omim_to_hpo.keys())

# 读取HPO_to_HPO关系文件
with open("../source_data/hpo_relations.txt", "r") as json_file:
    omim_to_hpo = json.load(json_file)
for i in range()
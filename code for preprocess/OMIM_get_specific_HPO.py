import json
from collections import defaultdict

# 加载每个OMIM对应的HPO的JSON文件
with open("../source_data/omim_to_hpo.json", "r") as json_file:
    omim_to_hpo = json.load(json_file)

# 加载每个HPO出现在不同疾病的次数的JSON文件
with open("../source_data/hpo_counts.json", "r") as json_file:
    hpo_counts = json.load(json_file)

# 创建一个字典来存储每个OMIM特异性的HPO
omim_specific_hpos = defaultdict(list)

# 遍历每个OMIM号对应的HPO列表
for omim, hpos in omim_to_hpo.items():
    # 对于每个HPO，检查其在其他疾病中出现的次数
    for hpo in hpos:
        count = hpo_counts.get(hpo, 0)
        # 如果HPO在其他疾病中出现的次数小于3，则将其添加到特异性HPO列表中
        if count < 3:
            omim_specific_hpos[omim].append(hpo)

# 确保所有OMIM都在结果中，并且没有特异性HPO的OMIM具有空值
for omim in omim_to_hpo.keys():
    if omim not in omim_specific_hpos:
        omim_specific_hpos[omim] = []

# 将特异性HPO字典保存为JSON文件
with open("../source_data/omim_specific_hpos.json", "w") as json_file:
    json.dump(omim_specific_hpos, json_file)

print("OMIM specific HPOs saved to omim_specific_hpos.json")

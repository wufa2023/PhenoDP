import json
from collections import defaultdict

# 读取包含 HPO 和 OMIM 关系的文本文件
with open("../source_data/hpo_to_omim_results.txt", "r") as file:
    lines = file.readlines()

# 创建一个字典来存储每个 HPO 出现在不同 OMIM 号中的次数
hpo_counts = defaultdict(int)

# 遍历每一行，更新 HPO 的出现次数
for line in lines:
    hpo, omim = line.strip().split(",")
    hpo_counts[hpo] += 1

# 将字典保存为 JSON 文件
with open("../source_data/hpo_counts.json", "w") as json_file:
    json.dump(hpo_counts, json_file)

print("HPO counts saved to hpo_counts.json")

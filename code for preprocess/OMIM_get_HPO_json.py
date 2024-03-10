import json
from collections import defaultdict

# 读取包含 HPO 和 OMIM 关系的文本文件
with open("../source_data/hpo_to_omim_results.txt", "r") as file:
    lines = file.readlines()

# 创建一个字典来存储每个 OMIM 号对应的 HPO 号列表
omim_to_hpo = defaultdict(list)

# 遍历每一行，将 HPO 和 OMIM 关系存储到字典中
for line in lines:
    hpo, omim = line.strip().split(",")
    omim_to_hpo[omim].append(hpo)

# 将字典保存为 JSON 文件
with open("../source_data/omim_to_hpo.json", "w") as json_file:
    json.dump(omim_to_hpo, json_file)

print("OMIM to HPO relationships saved to omim_to_hpo.json")

# to read omim to hpo json, you may need to run:
import json

# # 加载保存的 JSON 文件
# with open("omim_to_hpo.json", "r") as json_file:
#     omim_to_hpo = json.load(json_file)
#
# # 访问每个 OMIM 号对应的 HPO 号列表
# for omim, hpos in omim_to_hpo.items():
#     print(f"OMIM: {omim} -> HPOs: {', '.join(hpos)}")

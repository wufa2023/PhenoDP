from collections import defaultdict

# 读取包含 HPO 和 OMIM 关系的文本文件
with open("source_data/hpo_to_omim_results.txt", "r") as file:
    lines = file.readlines()

# 创建一个字典来存储每个 OMIM 号对应的 HPO 号列表
omim_to_hpo = defaultdict(list)

# 遍历每一行，将 HPO 和 OMIM 关系存储到字典中
for line in lines:
    hpo, omim = line.strip().split(",")
    omim_to_hpo[omim].append(hpo)

# 打印每个 OMIM 号对应的 HPO 号列表
for omim, hpos in omim_to_hpo.items():
    print(f"{omim} -> {', '.join(hpos)}")

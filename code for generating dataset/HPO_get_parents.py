import json
from collections import defaultdict

# 读取文件
file_path = '../source_data/hpo_relations.txt'  # 将'your_file.txt'替换为你的文件路径
with open(file_path, 'r') as file:
    lines = file.readlines()

hpo_dict = defaultdict(list)
current_hpo = None

state = 0
# 遍历文件内容
count = 0
for line in lines:
    line = line.strip()
    if line.startswith('term-HP:'):
        state = 0
        current_hpo = line.split('-')[-1]

    if line.startswith('Parents:'):
        state = 1
        continue

    if line.startswith('Children:'):
        state = 2

    if state == 1:
        parents_line = line
        parents = parents_line.split('-')[0][:-1]
        hpo_dict[current_hpo].append(parents)



# 将字典保存为JSON文件
output_json_path = '../source_data/hpo_parents.json'
with open(output_json_path, 'w') as json_file:
    json.dump(hpo_dict, json_file, indent=2)

print(f'JSON文件已保存至: {output_json_path}')

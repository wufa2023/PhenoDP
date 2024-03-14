import json
with open('../source_data/hpo_parents.json') as f:
    data = json.load(f)

print(len(data.keys()))
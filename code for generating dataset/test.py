import json
import numpy as np
import random
from utils import is_connected
file_path = '../source_data/omim_to_hpo.json'
with open(file_path, "r") as f:
    data = json.load(f)

hpo_list = []
omim_ids = list(data.keys())
for ids in omim_ids:
    hpo_list.extend(data[ids])
hpo_list = np.unique(hpo_list)

for ids in omim_ids:
    ref_hpo = set(data[ids])
    query_hpo = random.sample(list(set(hpo_list) - ref_hpo), 2)
    print(is_connected(ref_hpo, query_hpo))



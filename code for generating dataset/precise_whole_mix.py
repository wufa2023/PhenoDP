
# read data
import json
import pandas as pd
with open('../source_data/omim_to_hpo.json') as f:
    data = json.load(f)

with open('../source_data/hpo_counts.json') as f:
    hpo_list = json.load(f)
    hpo_list = list(hpo_list.keys())

with open("../source_data/hpo_parents.json", "r") as json_file:
    omim_parents = json.load(json_file)

import pandas as pd
df = pd.read_csv('../source_data/hpo_specific.csv')
specific_list = df['hpo'].values

save_omim = []
save_hpo = []
unique_omim = []
unique_hpo = []
omim_id = list(data.keys())

import numpy as np
import random
from utils import *
for _ in range(10):
    print('iter', _)
    for i in range(len(omim_id)):
        print(i)
        ref = data[omim_id[i]]
        temp = list(ref)
        # select noise, (0.1, 0.49) random add
        choice = range(10, 50, 1)
        add_len = (random.choice(choice) / 100) * len(ref)
        _noise_len = np.round(add_len, 0)
        noise_list = []
        while len(noise_list) < _noise_len:
            term = random.choice(hpo_list)
            if term not in ref and is_connected(ref, term) == False:
                noise_list.append(term)
            else:
                continue
        noise_seq = temp.extend(noise_list)

        ref = data[omim_id[i]]
        temp = list(ref)
        change_seq = get_change(temp, specific_list=specific_list, relation=omim_parents)

        final_seq = np.unique(change_seq, noise_seq)
        save_omim.append(omim_id[i])
        save_hpo.append(final_seq)

# remove duplicate
print('generate records', len(save_omim))
unique_omim, unique_hpo = remove_duplicate(save_omim_id=save_omim, save_sim_hpo=save_hpo)
print('unique records', len(unique_omim))
pd.DataFrame({'id': unique_omim, 'sim_hpo': unique_hpo}).to_csv("../source_data/precise_whole_mix.csv", index=False)


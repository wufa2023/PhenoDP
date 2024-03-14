import json
import numpy as np
import pandas as pd
with open('../source_data/omim_specific_hpos.json', 'r') as f:
    data = json.load(f)

hpo_list = []
for key, values in data.items():
    hpo_list.extend(values)

pd.DataFrame({'hpo':np.unique(hpo_list)}).to_csv('../source_data/hpo_specific.csv', index=False)

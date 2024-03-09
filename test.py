import requests
import pandas as pd

url = "https://hpo.jax.org/api/hpo/term/HP:0000002/diseases"
response = requests.get(url)
print(1)
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data['diseases'])

    # Filter rows where 'db' column is 'OMIM'
    omim_df = df.loc[df['db'] == 'OMIM']

    print(omim_df)
else:
    print("Failed to retrieve data. Status code:", response.status_code)

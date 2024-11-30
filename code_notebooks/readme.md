## Recommended Usage

We recommend that you download the data files directly from the homepage and then refer to the tutorial code in `tutorial.ipynb` for usage.

### File Descriptions:
- **`PCL_HPOEncoder.py`** and **`PSD_HPOEncoder.py`**: These are the corresponding model code files, implemented using PyTorch.
- **`PhenoDP.py`**: This is the main tool operation class code.
- **`PhenoDP_Preprocess.py`**: This is a preprocessing utility class code.

### For Developers:
You can refer to the training notebooks:
- `PSD-HPOEncoder`
- `PCL-HPOEncoder`
- `Summarizer's training.ipynb`

These notebooks contain the default parameters, which you can adjust according to your needs.

### Preprocessing for Faster Computation:
To accelerate the computation, we have precomputed the similarity between HPO terms and diseases listed in OMIM using the JC method. The code for this process can be found in `preprocess_similarity_file.ipynb`.

### System Environment:
The **`pip_list`** records the environment we used to run PhenoDP, which has been tested on:
- **Distributor ID**: Ubuntu
- **Description**: Ubuntu 18.04.6 LTS
- **Release**: 18.04

### Hardware:
Our experimental model was trained on a **GeForce RTX 3090** GPU.

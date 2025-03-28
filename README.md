# PhenoDP Latest Version Description

In this GitHub repository, we provide the files involved in training and using PhenoDP, as well as a demonstration of how to use the latest version of PhenoDP. For details on the PhenoDP code files, please refer to the `code_in_version_2` folder. The `data` folder contains synthetic cases generated by DeepSeek-R1-671B, which serve as training texts for the summarizer. We also provide disease entries from OMIM or Orphanet, HPO annotations, and HPO definition sentences. Additionally, the three datasets involved in the benchmark are included, along with an independent text evaluation dataset, SUMPUBMED, of which we used 1,000 entries.

## Preprocessing Files for Ranker and Recommender
To run the Ranker and Recommender, four preprocessing files are required:

1. **HPO Annotation Files** (`hp.obo`, `phenotype.hpoa`, `phenotype_to_gene.txt`):  
   These files are used to update the Ontology class from PyHPO, a utility class for quick reading of HPO annotations. If not specified otherwise, the default annotation files in the package directory will be used. (Optional)

2. **JC Similarity Matrix** (`JC_sim_dict.pkl`):  
   This file contains the Jaccard-Cosine (JC) similarity matrix between HPO terms and each disease. It is generated in the `Get_Similarity_Matrix.ipynb` notebook.

3. **HPO Semantic Embedding Matrix** (`node_embedding_dict.pkl`):  
   This matrix is generated by the summarizer and PSD-HPOEncoder. For details, refer to the `Node_Embedding_Generation.ipynb` notebook.

4. **Transformer Encoder Weights** (`transformer_encoder_infoNCE.pth`):  
   This file contains the parameter weights of the recommender after contrastive learning training. It is generated in the `Train_PCL-HPOEncoder.ipynb` notebook. The process can be implemented using single-threading (which typically takes several hours) or multi-threading (which is faster). For details, refer to the `Get_Similarity_Matrix.ipynb` notebook, which demonstrates how to generate the latest version of the similarity matrix using the most recent HPO annotation files.

## Usage Instructions
Please directly download the provided files in `code_in_version_2`: `PCLHPOEncoder.py`, `PSD_HPOEncoder.py`, `PhenoDP.py`, and `PhenoDP_Prepocess.py`. Place them in a folder, and then you can directly import the contents of these four code files in a Jupyter notebook using `from PhenoDP import *`. We believe this approach is simpler and more straightforward.

To use PhenoDP, refer to the following notebooks:  
- `Run_Ranker_Recommender.ipynb`: For running the Ranker and Recommender.  
- `Use the summarizer..ipynb`: For generating summaries.

## Environment for Ranker and Recommender
- **Python**: 3.7.3  
- **pyhpo**: 3.3.1  
- **torch**: 1.9.0+cu111  
- **torchaudio**: 0.9.0  
- **torchvision**: 0.10.0+cu111  
- **transformers**: 4.45.0  
- **dgl**: 0.9.1  
- **networkx**: 3.1  
- **tqdm**: 4.66.5  
- **pandas**: 1.5.3  
- **numpy**: 1.24.1  

## Environment for Summarizer
- **torch**: 2.0.1+cu118  
- **tokenizers**: 0.20.1  
- **peft**: 0.13.2  

## Key Updates in the Latest Version
In the latest version, we have separated the Python environments for the summarizer and the Ranker/Recommender. This decision was made primarily because the summarizer can utilize more advanced language models to ensure enhanced functionality. This inevitably requires the use of more advanced versions of Torch. However, the Ranker and Recommender, which are based on Torch 1.9.0, may encounter errors when using these more advanced Torch versions. To avoid such errors, we have maintained Torch 1.9.0 for the Ranker and Recommender, while allowing the summarizer to use any advanced version of Torch.

We have provided the four preprocessed files and the model weight files directly in the link below:  
https://drive.google.com/drive/u/0/folders/1S6ZJC-5YaM18o7D0sjJ3Ae_w5jO_bMBt

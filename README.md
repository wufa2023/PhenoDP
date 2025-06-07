# PhenoDP: Leveraging Deep Learning for Phenotype-Based Case Reporting, Disease Ranking, and Symptom Recommendation


PhenoDP is an advanced toolkit for phenotype-driven disease diagnosis and prioritization using Human Phenotype Ontology (HPO) data. It offers a powerful **Summarizer** for clinical summaries, a **Ranker** for disease prioritization, and a **Recommender** for HPO term suggestion.

![phenodp framework](data/phenodp.jpg)

## Features

- **Summarizer**: Utilizes a distilled Bio-Medical-3B-CoT model to generate high-quality, patient-centered clinical summaries from HPO terms, enhancing the interpretability of symptoms.
- **Ranker**: Integrates IC values, Phi coefficients, and Graph Convolutional Networks (GCN) for precise disease ranking, excelling particularly in complex disease scenarios.
- **Recommender**: Employs a Transformer model optimized with contrastive learning to intelligently suggest critical symptoms for distinguishing diseases, improving diagnostic accuracy and confidence.

## Installation

### Dependencies Installation

```bash
# Create environment
conda create -n phenodp python=3.10 -y
conda activate phenodp

# Install PyTorch with CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu118

# Install DGL with CUDA 11.8 support (Please check your torch and CUDA version)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html

# Install other dependencies
pip install transformers pandas numpy tqdm scipy obonet networkx "pyhpo==3.3.2" accelerate
```

### Install PhenoDP

```bash
git clone https://github.com/yourusername/phenodp.git
cd phenodp
pip install -e .
```

### Required Data Files

PhenoDP requires preprocessed data files to function. After installation, you can either download the preprocessed data or train the models yourself.

```bash
# Download files from Google Drive
# https://drive.google.com/drive/u/0/folders/1S6ZJC-5YaM18o7D0sjJ3Ae_w5jO_bMBt
```

Place the downloaded files in the data/ directory:
- `JC_sim_dict.pkl` - JC similarity matrix 
- `node_embedding_dict.pkl` - HPO semantic embeddings   
- `transformer_encoder_infoNCE.pth` - Transformer weights

Additionally, you can download the latest version of HPO, which requires downloading the files `hp.obo`, `hp.json`, `hp.owl`, `genes_to_disease.txt`, `genes_to_phenotype.txt`, `phenotype_to_genes.txt`, and `phenotype.hpoa`. Store these files in a folder (it is recommended to create a subfolder like hpo_latest within the data/ directory for unified storage, facilitating subsequent data processing and model training calls).


## Quick Start

Please see `notebooks/Tutorial.ipynb`


## Contact Us

If you encounter any issues during use or have any suggestions, feel free to contact us:

- Email: blwen24@m.fudan.edu.cn
- Email: ylong2025@163.com

You can also submit an issue on GitHub.

## Citation

If you use PhenoDP in your research, please cite:

```bibtex
@article{wen2025phenodp,
  title={PhenoDP: leveraging deep learning for phenotype-based case reporting, disease ranking, and symptom recommendation},
  author={Wen, B. and Shi, S. and Long, Y. and others},
  journal={Genome Medicine},
  volume={17},
  pages={67},
  year={2025},
  doi={10.1186/s13073-025-01496-8},
  url={https://doi.org/10.1186/s13073-025-01496-8}
}
```
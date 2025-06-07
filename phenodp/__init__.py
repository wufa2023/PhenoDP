"""
PhenoDP: Leveraging Deep Learning for Phenotype-Based Case Reporting, Disease Ranking, and Symptom Recommendation

PhenoDP is an advanced toolkit for phenotype-driven disease diagnosis and prioritization using Human Phenotype Ontology (HPO) data. It offers a powerful **Summarizer** for clinical summaries, a **Ranker** for disease prioritization, and a **Recommender** for HPO term suggestion.
"""

__version__ = "2.0.0"
__author__ = "Tianlab-Bioinfo"
__email__ = "blwen24@m.fudan.edu.cn"

# Import main classes
from .core import PhenoDP
from .preprocess import PhenoDP_Initial
from .encoders import PCL_HPOEncoder, PSD_HPOEncoder

# Import utility functions
from .utils import *

__all__ = [
    "PhenoDP",
    "PhenoDP_Initial", 
    "PCL_HPOEncoder",
    "PSD_HPOEncoder",
] 
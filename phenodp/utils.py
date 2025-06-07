"""
Utility functions for PhenoDP

This module contains helper functions for data loading, preprocessing,
and other common operations.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import pkg_resources


def load_pickle(file_path: str) -> object:
    """Load pickle file safely"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading pickle file {file_path}: {str(e)}")


def save_pickle(obj: object, file_path: str) -> None:
    """Save object to pickle file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise RuntimeError(f"Error saving pickle file {file_path}: {str(e)}")


def get_package_data_path(filename: str) -> str:
    """Get path to data file in package"""
    try:
        return pkg_resources.resource_filename('phenodp', f'data/{filename}')
    except:
        # Fallback for development
        return os.path.join(os.path.dirname(__file__), 'data', filename)


def validate_hpo_terms(hpo_terms: List[str]) -> List[str]:
    """Validate and filter HPO terms"""
    valid_terms = []
    for term in hpo_terms:
        if isinstance(term, str) and term.startswith('HP:'):
            valid_terms.append(term)
        else:
            print(f"Warning: Invalid HPO term format: {term}")
    return valid_terms


def load_similarity_matrix(file_path: Optional[str] = None) -> Dict:
    """Load similarity matrix from file or package data"""
    if file_path is None:
        file_path = get_package_data_path('JC_sim_dict.pkl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Similarity matrix file not found: {file_path}\n"
            "Please download the preprocessed files from: "
            "https://drive.google.com/drive/u/0/folders/1S6ZJC-5YaM18o7D0sjJ3Ae_w5jO_bMBt"
        )
    
    return load_pickle(file_path)


def load_node_embeddings(file_path: Optional[str] = None) -> Dict:
    """Load node embeddings from file or package data"""
    if file_path is None:
        file_path = get_package_data_path('node_embedding_dict.pkl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Node embeddings file not found: {file_path}\n"
            "Please download the preprocessed files from: "
            "https://drive.google.com/drive/u/0/folders/1S6ZJC-5YaM18o7D0sjJ3Ae_w5jO_bMBt"
        )
    
    return load_pickle(file_path)


def check_required_files() -> Dict[str, bool]:
    """Check if all required preprocessed files are available"""
    required_files = {
        'JC_sim_dict.pkl': 'JC similarity matrix',
        'node_embedding_dict.pkl': 'HPO semantic embeddings',
        'transformer_encoder_infoNCE.pth': 'Transformer encoder weights'
    }
    
    file_status = {}
    for filename, description in required_files.items():
        file_path = get_package_data_path(filename)
        file_status[description] = os.path.exists(file_path)
    
    return file_status


def print_file_status():
    """Print status of required files"""
    status = check_required_files()
    print("Required file status:")
    print("-" * 50)
    for description, exists in status.items():
        status_str = "✓ Available" if exists else "✗ Missing"
        print(f"{description}: {status_str}")
    
    if not all(status.values()):
        print("\nMissing files can be downloaded from:")
        print("https://drive.google.com/drive/u/0/folders/1S6ZJC-5YaM18o7D0sjJ3Ae_w5jO_bMBt")


def format_results(results: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Format and limit results DataFrame"""
    if len(results) > top_n:
        results = results.head(top_n)
    
    # Round similarity scores for better display
    if 'Similarity' in results.columns:
        results['Similarity'] = results['Similarity'].round(4)
    
    return results.reset_index(drop=True)


def save_results(results: pd.DataFrame, output_path: str, format: str = 'csv') -> None:
    """Save results to file"""
    try:
        if format.lower() == 'csv':
            results.to_csv(output_path, index=False)
        elif format.lower() == 'excel':
            results.to_excel(output_path, index=False)
        elif format.lower() == 'json':
            results.to_json(output_path, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        print(f"Results saved to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving results: {str(e)}")


__all__ = [
    'load_pickle',
    'save_pickle', 
    'get_package_data_path',
    'validate_hpo_terms',
    'load_similarity_matrix',
    'load_node_embeddings',
    'check_required_files',
    'print_file_status',
    'format_results',
    'save_results'
] 
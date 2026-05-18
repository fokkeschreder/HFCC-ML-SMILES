# HFCC ML SMILES

A machine learning model trained for predicting isotropic proton hyperfine coupling constants (HFCC) in hydrocarbon radicals directly from SMILES strings using Graph Convolutional Networks (GCN).

## Overview

1. **Dataset Generation**: Creating 3D geometries of alkyl and alkenyl radicals from SMILES strings using RDKit's ETKDG algorithm.
2. **Quantum Chemistry Calculations**: Generating ORCA input files for geometry optimization (wB97X-D/6-31G*) and EPR property calculations (wB97X-D3/IGLO-II).
3. **Data Extraction**: Extracting and averaging the calculated HFCC values from ORCA output files.
4. **Machine Learning**: Training a PyTorch Geometric (PyG) Graph Convolutional Network (GCN) model to predict these constants directly from the 2D molecular graphs.

## Repository Structure

### Dataset Generation & Quantum Chemistry
*   `generate_alkyl.py` / `generate_alkenyl.py`: Scripts to generate RDKit structures and ORCA input files for alkyl, cycloalkyl, and alkenyl radicals.
*   `extract_hfcc.py`: Parses ORCA output files to extract isotropic hyperfine coupling constants for equivalent protons.
*   `generate_epr_database.py`: Aggregates the extracted data into the final structured dataset (`epr_dataset.csv`).

### Machine Learning Pipeline
*   `dataset.py`: PyTorch Geometric `InMemoryDataset` implementation that converts SMILES and HFCC targets into graph representations, featurizing nodes (atoms) and edges (bonds).
*   `model.py`: GCN-based deep learning architecture for atom-level property prediction.
*   `train.py`: Training loop for the PyTorch model, handling data splitting, loss calculation, and model saving.
*   `predict.py`: Inference script for predicting HFCCs of new molecules using a trained model.

## Requirements

*   `torch`
*   `torch_geometric`
*   `rdkit`
*   `pandas`
*   `matplotlib`

"""Functions for training models using Pycaret"""

import pandas as pd
import numpy as np
from pycaret.regression import (
    set_config,
    setup,
    compare_models,
    tune_model,
    finalize_model,
)
from rdkit.Chem import PandasTools
from solubility_model.utils.paths import make_dir_function as mdf


def train_rdkit_models():
    """Trains 25 ML models using PyCaret Vernalis solubility data.
    Uses RDKit 2D descriptors and Morgan count fingerprint.
    Selects the top 4 best performing and tunes and finalizes them.

    Returns:
        List: The top 4 finalized pipelines from Pycaret
    """
    # Load 120 time point data and descriptors
    train_desc = pd.read_csv(mdf("data/processed/train_descriptors.csv")())
    train_data = pd.read_csv(mdf("data/processed/train_smiles.csv")())
    # Generate Murcko scaffold for splitting data
    PandasTools.AddMoleculeColumnToFrame(train_data, smilesCol="smiles")
    PandasTools.AddMurckoToFrame(train_data)
    scaffolds = train_data.Murcko_SMILES
    # Train models using Pycaret and select top 4
    all_data = pd.concat([train_data[["y_true"]], train_desc], axis=1)
    _ = setup(
        all_data,
        target="y_true",
        normalize=True,
        transformation=True,
        feature_selection=True,
        fold_strategy="groupkfold",
        fold_groups=scaffolds,
        verbose=False,
    )
    set_config("seed", 42)
    top_4_models = compare_models(n_select=4, round=3, verbose=False)
    # Tune and finalize 4 models
    tuned_4 = [tune_model(x, verbose=False) for x in top_4_models]
    final_4 = [finalize_model(x) for x in tuned_4]
    return final_4

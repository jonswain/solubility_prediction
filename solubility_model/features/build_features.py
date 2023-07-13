"""Functions for generating features for classical ML models."""

import pandas as pd
from molfeat.calc import FPCalculator, get_calculator
from molfeat.trans import MoleculeTransformer
from solubility_model.utils.paths import make_dir_function as mdf


def all_descriptors(dataframe):
    """Generates Morgan count fingerprints and all RDKit 2D descriptors.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe with a column of canonical smiles.

    Returns:
        Pandas dataframe: The combined features dataframe.
    """
    params = {
        "radius": 3,
        "nBits": 2048,
        "useChirality": True,
        "useFeatures": True,
    }

    fp_calc = FPCalculator("ecfp-count", **params)
    fp_transf = MoleculeTransformer(fp_calc, n_jobs=-1)

    rdkit_calc = get_calculator("desc2d")
    rdkit_transf = MoleculeTransformer(rdkit_calc, n_jobs=-1)

    df_desc = pd.DataFrame(fp_transf(dataframe.smiles), columns=fp_calc.columns)
    df_rdkit = pd.DataFrame(rdkit_transf(dataframe.smiles), columns=rdkit_calc.columns)
    df_rdkit.drop("Alerts", axis=1, inplace=True)

    return pd.concat([df_desc, df_rdkit], axis=1)


def generate_descriptors():
    """Takes the training and test data sets and generates features.
    Saves the data to the data/processed/ directory.
    """
    for dataset in ["train", "test"]:
        data = pd.read_csv(mdf(f"data/processed/{dataset}_smiles.csv")())
        descriptors = all_descriptors(data)
        descriptors.to_csv(
            mdf(f"data/processed/{dataset}_descriptors.csv")(),
            index=False,
        )

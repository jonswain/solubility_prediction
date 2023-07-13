"""Functions to make dataset for solubility project"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import CanonSmiles
from rdkit.Chem.MolStandardize.rdMolStandardize import CanonicalTautomer
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from solubility_model.utils.paths import make_dir_function as mdf


def process_data():
    """Processes useful columns from the provided data and saves to data/interim/"""

    # Open file
    raw_data = pd.read_excel(mdf("data/external/solubility_data.xlsx")())
    filtered_data = raw_data[["SMILES", "LogS.M."]]

    # Add Mol col, standardize tautomers, generate InChiKeys
    PandasTools.AddMoleculeColumnToFrame(filtered_data, smilesCol="SMILES")
    filtered_data = filtered_data.dropna()
    filtered_data["standardized_mol"] = filtered_data.ROMol.apply(CanonicalTautomer)
    filtered_data["InChiKey"] = filtered_data.standardized_mol.apply(
        Chem.inchi.MolToInchiKey
    )

    # Average repeats based on InChiKeys
    aggregated_df = (
        filtered_data[["InChiKey", "LogS.M."]].groupby("InChiKey").agg("mean")
    )
    smiles_dict = filtered_data[["InChiKey", "SMILES"]].groupby("InChiKey").first()
    merged_df = aggregated_df.merge(right=smiles_dict, on="InChiKey").reset_index()

    # Tidy and save data
    merged_df.columns = ["InChiKey", "y_true", "smiles"]
    merged_df.smiles = merged_df.smiles.apply(CanonSmiles)
    merged_df[["smiles", "y_true"]].to_csv(
        mdf("data/interim/processed_dataset.csv")(), index=False
    )


def train_test_split(train_frac=0.8, how="time"):
    """Splits the dataset into training and test fractions using the method specified.

    Args:
        train_frac (float, optional): Proportion of dataset to use as test data.
        Defaults to 0.8.
        how (str): Method for split. ["time", "scaffold", "random"]
    """
    all_data = pd.read_csv(mdf("data/interim/processed_dataset.csv")())

    if how == "time":
        div = int(len(all_data) * train_frac)
        all_data.iloc[div:].to_csv(
            mdf("data/processed/train_smiles.csv")(), index=False
        )
        all_data.iloc[:div].to_csv(mdf("data/processed/test_smiles.csv")(), index=False)

    elif how == "random":
        train_data = all_data.sample(frac=train_frac)
        test_data = all_data.drop(train_data.index)
        train_data.to_csv(mdf("data/processed/train_smiles.csv")(), index=False)
        test_data.to_csv(mdf("data/processed/test_smiles.csv")(), index=False)

    else:
        mols = [Chem.MolFromSmiles(x) for x in all_data.smiles.to_list()]
        scaffolds = [Chem.MolToSmiles(GetScaffoldForMol(mol)) for mol in mols]
        # Split based on scaffold
        gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=0)
        train_index, test_index = list(gss.split(all_data.smiles, groups=scaffolds))[0]
        train_data = all_data.iloc[train_index].reset_index(drop=True)
        test_data = all_data.iloc[test_index].reset_index(drop=True)
        assert (
            len(
                ({scaffolds[i] for i in set(train_index)}).intersection(
                    ({scaffolds[i] for i in set(test_index)})
                )
            )
            == 0
        )
        # Save data
        train_data.to_csv(mdf("data/processed/train_smiles.csv")(), index=False)
        test_data.to_csv(mdf("data/processed/test_smiles.csv")(), index=False)

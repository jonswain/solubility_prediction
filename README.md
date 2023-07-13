# solubility_model

Exploring ML models to predict solubility of small organic compounds, trained on data provided by the AI4SD summer school 2022.
  
## Installation guide

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)
- Optional [Mamba](https://mamba.readthedocs.io/en/latest/)

### Create environment

```bash
conda env create -f environment.yml
activate solubility_model
```

or 

```bash
mamba env create -f environment.yml
activate solubility_model
```

The packages necessary to run the project are now installed inside the conda environment.

**Note: The following sections assume you are located in your conda environment.**

### Set up project's module

To move beyond notebook prototyping, all reusable code should go into the `solubility_model/` folder package. To use that package inside your project, install the project's module in editable mode, so you can edit files in the `solubility_model` folder and use the modules inside your notebooks :

```bash
pip install --editable .
```

To use the module inside your notebooks, add `%autoreload` at the top of your notebook :

```python
%load_ext autoreload
%autoreload 2
```

Example of module usage :

```python
from solubility_model.utils.paths import data_dir
data_dir()
```

## Project Organization

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures         <- Generated graphics and figures to be used in reporting.
    │
    ├── environment.yml    <- The requirements file for reproducing the analysis environment.
    │
    ├── .here              <- File that will stop the search if none of the other criteria
    │                         apply when searching head of project.
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .)
    │                         so solubility_model can be imported.
    │
    └── solubility_model               <- Source code for use in this project.
        ├── __init__.py    <- Makes solubility_model a Python module.
        │
        ├── data           <- Scripts to download or generate data.
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions.
        │   ├── predict_model.py
        │   └── train_model.py
        │
        ├── utils          <- Scripts to help with common tasks.
        |   └── paths.py   <- Helper functions to relative file referencing across project.
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations.
            └── visualize.py

---

Project based on the [cookiecutter data science project template](https://github.com/jonswain/cookie-cutter-data-science).

Made by Jon Swain, contact at jonswain123@gmail.com.
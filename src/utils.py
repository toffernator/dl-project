from pathlib import Path

import pandas as pd


def get_dataset_name(file_name_with_dir: Path) -> str:
    file_name_without_dir = file_name_with_dir.name
    name_parts = file_name_without_dir.split("_")
    dataset_name = "_".join(name_parts[:-1])
    return dataset_name

def conf_matrix_to_latex_table(conf_matrix, subject, model_name):
    table_path_root: Path = Path(".ignore") / Path ("confusion_matrices") / Path(model_name)
    save_to: Path = table_path_root / Path(f"{subject}.tex")
    pd.DataFrame(conf_matrix).to_latex(save_to)

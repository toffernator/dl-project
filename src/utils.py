from pathlib import Path

def get_dataset_name(file_name_with_dir: Path) -> str:
    file_name_without_dir = file_name_with_dir.name
    name_parts = file_name_without_dir.split("_")
    dataset_name = "_".join(name_parts[:-1])
    return dataset_name
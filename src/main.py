from pathlib import Path

import h5py


def get_dataset_name(file_name_with_dir: Path) -> str:
    file_name_without_dir = file_name_with_dir.name
    name_parts = file_name_without_dir.split("_")
    dataset_name = "_".join(name_parts[:-1])
    return dataset_name


def main():
    filename = Path("dataset/Intra/train/rest_105923_1.h5")
    with h5py.File(filename, "r") as f:
        dataset_name = get_dataset_name(filename)
        matrix = f.get(dataset_name)[()]

        print(type(matrix))
        print(matrix.shape)


if __name__ == "__main__":
    main()

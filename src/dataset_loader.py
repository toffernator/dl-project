from pathlib import Path
import h5py
import numpy as np
from enum import Enum

DATASET_DIR = "dataset"
INTRA_DIR = "Intra"
CROSS_DIR = "Cross"
PREPROCESSED_DIR = "Preprocessed"


class Label(Enum):
    rest = 0
    motor = 1
    math = 2
    memory = 3

    @staticmethod
    def from_str(label):
        match label:
            case "rest":
                return Label.rest
            case "motor":
                return Label.motor
            case "memory":
                return Label.memory
            case "math":
                return Label.math

        raise RuntimeError(f"Unknown label {label}")

    def __str__(self):
        return self.name


class DatasetFile:
    path: Path
    name: str
    subject: str
    label: Label
    preprocessed: bool

    def __init__(self, filepath: Path, load_preprocess=True):
        self.path = filepath
        self.name, self.subject = get_dataset_name_subject(filepath)
        self.label = label_dataset(filepath)
        self.preprocessed = False

        if not load_preprocess:
            return

        preprocessed_path = get_preprocessed_path(filepath)
        if preprocessed_path.exists():
            self.path = preprocessed_path
            self.preprocessed = True

    def load(self) -> np.ndarray:
        if self.preprocessed:
            with open(self.path, "rb") as f:
                matrix = np.load(f)
                return matrix
        else:
            with h5py.File(self.path, "r") as f:
                matrix = f.get(self.name)[()]
                return matrix  # orig size: (248, 35624)

    def save_preprocessed(self, matrix: np.ndarray):
        self.path = save_preprocessed(self.path, matrix)
        self.preprocessed = True

    def __str__(self):
        return f"<DatasetFile '{str(self.path)}' label={self.label}({int(self.label.value)}) pp={self.preprocessed}>"

    def __repr__(self):
        return str(self)


# returns a pair of list of training dataset files and list of test dataset files from "Intra"
def get_intra_dataset_files(load_preprocess=True):
    return get_dataset_files(INTRA_DIR, load_preprocess)


# returns a pair of list of training dataset files and list of test dataset files from "Cross"
def get_cross_dataset_files(load_preprocess=True):
    return get_dataset_files(CROSS_DIR, load_preprocess)


def save_preprocessed(original_path: Path, matrix: np.ndarray):
    path = get_preprocessed_path(original_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, matrix)

    return path


### Helpers


def get_dataset_files(training_type, load_preprocess=True):
    path = Path(f"{DATASET_DIR}/{training_type}")
    if not (path.exists() and path.is_dir()):
        raise FileNotFoundError(f"{training_type} dataset not found")

    train = [DatasetFile(f, load_preprocess) for f in path.glob("train/*.h5")]
    test = [DatasetFile(f, load_preprocess) for f in path.glob("test*/*.h5")]

    return (train, test)


def get_dataset_name_subject(filepath: Path):
    filename = filepath.name
    temp = filename.split("_")[:-1]
    subject = temp[-1]
    dataset_name = "_".join(temp)
    return (dataset_name, subject)


def get_preprocessed_path(original_path: Path):
    pathstr = str(original_path)
    if pathstr.startswith(f"{DATASET_DIR}/{PREPROCESSED_DIR}"):
        return original_path

    preprocessed_path = f"{DATASET_DIR}/{PREPROCESSED_DIR}/{pathstr[8:-3]}.npy"
    return Path(preprocessed_path)


def label_dataset(filepath: Path):
    filename = filepath.name
    for label in Label:
        try:
            filename.index(label.name)
            return label
        except:
            pass

    raise RuntimeError(f"Unknown label for dataset {filename}")

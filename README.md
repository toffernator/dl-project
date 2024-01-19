# DL Project

## Setting up

### Dataset

First get the dataset placed in the correct place:

1. Copy the provided dataset file `Final Project data.zip` to the main directory
2. Run `setup_dataset.sh`, or
3. Unzip the zip file and rename the folder `Final Project data` to `dataset`

Your final directory structure should look like:

```
.
├── dataset
│   ├── Cross
│   │   ├── test1
│   │   ├── test2
│   │   ├── test3
│   │   └── train
│   └── Intra
│       ├── test
│       └── train
├── docs
└── src
```

### Dependencies

The program has been developed using Python 3.11, you can install package dependencies using `pip`:

```sh
pip install scipy scikit-learn h5py matplotlib tensorflow keras pandas
```

## Running

The python program can be run as a module with the command

```sh
python -m src.main
```

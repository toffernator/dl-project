# DL Project

## Setting up

First get the dataset placed in the correct place:

1. Copy the provided dataset file `Final Project data.zip` to the main directory
2. Run `setup_dataset.sh`, or
3. Unzip the zip file and rename the folder `Final Project data` to `dataset`

Next, make sure that [Poetry](https://python-poetry.org/docs/#installation) is installed ([pipx](https://pipx.pypa.io/latest/installation/):

```sh
pipx install poetry==1.7.1
```

Then you run to get started

```sh
poetry install
poetry run main
```

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


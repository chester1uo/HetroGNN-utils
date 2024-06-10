# Graph Construction from Edge Lists and Node Features

This repository provides a Python script to construct a heterogeneous graph from edge lists and node features, using multiprocessing to optimize performance. The script is designed to work with the IEEE-CIS Fraud Detection dataset but can be adapted for other datasets.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [License](#license)

## Overview

This project includes a script to construct a graph using edge lists and node features from a dataset. The graph construction process leverages multiprocessing to speed up the handling of large datasets. The script reads a configuration file in JSON format to specify the input files and parameters.

## Requirements

- Python 3.7+
- DGL (Deep Graph Library)
- NumPy
- Pandas
- PyTorch
- Multiprocessing

Install the required Python packages using pip:

```bash
pip install dgl numpy pandas torch
```

## Dataset

### IEEE-CIS Fraud Detection

The IEEE-CIS Fraud Detection dataset is used as an example in this project. The dataset consists of transaction data and identity data. It can be downloaded from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data).

### Data Structure

- `train_transaction.csv`: Transaction data including features like `TransactionID`, `isFraud`, `TransactionDT`, etc.
- `train_identity.csv`: Identity data with features like `TransactionID`, `DeviceType`, `id_01`, etc.
- `joint_data.csv`: Joint data of identity and transaction data.

This project can not only be applied on this dataset, it can process data with similar structure.

## Configuration

The script uses a JSON configuration file to specify input files and parameters. Below is an example configuration file (`config.json`):

```json
{
    "input_data": "./data/joint_data.csv",
    "edgelist_expression": "relation*",
    "output_dir": "./output",
    "node_cols": ["isFraud", "TransactionDT", "card1", "card2", ...],
    "key_col": "TransactionID",
    "feature_cols": ["TransactionID", "TransactionAmt", "dist1", "dist2", "C1", "C2", "C3", "C4", "C5", "C6", "C7", ...],
    "num_workers": 4
}
```

### Configuration Parameters

- `input_data`: Path to the CSV file containing the transaction data.
- `edgelist_expression`: Regular expression to match edge list files.
- `output_dir`: Directory where output files will be saved.
- `node_cols`: List of columns to be used for generating edge lists.
- `key_col`: Column to be used as the key for generating edges.
- `feature_cols`: List of columns to be used as node features.
- `num_workers`: Number of worker processes for parallel processing.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-repo/graph-construction.git
cd graph-construction
```

2. Place your dataset in the `data` directory and update the `config.json` file with the appropriate paths and parameters.

3. Run the script:

```bash
python construct_graph.py config.json
```

The script will read the configuration file, process the dataset, and construct the graph.

## Output

The output directory will contain:
- `relation_*_edgelist.csv`: Edge list files for each specified column.
- `feature.csv`: Node feature file.
- `graph_data.pth`: Serialized graph data including the graph, features, and node mappings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

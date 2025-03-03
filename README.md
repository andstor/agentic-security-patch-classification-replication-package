# XXXX 2025 replication package


## Repository structure

This repository is organized as follows:
- **/data**: Contains the experiments' generated data.
- **/analysis**: Contains all scripts used for data analysis.
- **/figures**: Contains all figures created during data analysis.

## Data
```
data/
|-- TODO
```


## Analysis
From the generated data, we fix it using the `fix_data.ipynb` notebook. After fixing the data, we calculate the CodeBLEU scores using the `calc_score.ipynb` notebook. Finally, we analyze the data and generate the plots using the `plots.ipynb` notebook.
```
analysis/
|-- plots.ipynb                  Jupyter Notebook file containing the Python code used to analyze the extracted data and generate the resulting plots.
```

## Replication
Follow the setup instructions within each directory. To replicate the experiments, each follow the steps below:

1. TODO
2. Analyze the data and generate the plots using the `plots.ipynb` notebook.
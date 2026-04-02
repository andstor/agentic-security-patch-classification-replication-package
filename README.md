# Agentic security patch classification replication package


## Repository structure

This repository is organized as follows:
- **/data**: Contains the experiments' source and generated data.
- **/analysis**: Contains all scripts used for data analysis.
- **/baselines**: Contains the replication material for each baseline.
- **/figures**: Contains all figures created during data analysis.
- **/evaluation**: Contains the scripts for evaluating Favia.
- **/tables**: Contains all tables created during data analysis.


## Data
```
data/
|-- analysis/                       The failure mode classification data.
|-- baselines/                      The data generated from replicating the baselines.
|   |-- PatchFinder/                The data generated from replicating PatchFinder.
|   |-- VulFixMiner/                The data generated from replicating VulFixMiner.
|   |-- LLM4VFD/                    The data generated from replicating LLM4VFD.
|   |-- CommitShield/               The data generated from replicating CommitShield.
|-- ground_truth/                   The ground truth data for the CVEs and commits.
|   |--cve_github_can.pkl           Data from  http://dx.doi.org/10.1145/3597503.3639202
|-- output/                         Contains the output data from evaluating Favia.
|   |<subset>_<model_name>.jsonl  The output data from evaluating Favia on a specific subset and model.
|   |<subset>_<model_name>.tar    Tar file of pickled smolagents RunResult dumps of data from evaluating Favia on a specific subset and model.
|-- traces/                           Contains the execution traces collected from evaluating Favia.
|   |<subset>_<model_name>.jsonl.zip  The execution traces from evaluating Favia on a specific subset and model. Very large, so zipped.
|-- results_agent.csv              CSV file containing the results of evaluating Favia on the different subsets and models.
|-- results_PatchFinder.csv        CSV file containing the evaluation results of PatchFinder.
|-- results_VulFixMiner.csv        CSV file containing the evaluation results of VulFixMiner.
|-- results_LLM4VFD.csv            CSV file containing the evaluation results of LLM4VFD.
|-- results_CommitShield.csv       CSV file containing the evaluation results of CommitShield.
|-- token_usage_agent.csv          CSV file containing the token usage of the agent.
|-- token_usage_LLM4VFD.csv        CSV file containing the token usage of LLM4VFD.
|-- token_usage_CommitShield.csv   CSV file containing the token usage of CommitShield.
```

## Analysis
We analyze the data and generate the plots using the `plots.ipynb` notebook.
```
analysis/
|-- plots.ipynb                  Jupyter Notebook file containing the Python code used to analyze the extracted data and generate the resulting plots.
```

## Replication
Follow the setup instructions within each directory. To replicate the experiments, each follow the steps below:

1. Prepare base dataset. Run `data.ipynb` in `notebooks/` to prepare the base dataset.
2. Package the `cvevc_cve` dataset by running `data_cve.py` in `notebooks/`.
3. Scrape GitHub repositories and extract the data. See README in `scraping/` for instructions. Produces the `cvevc_commits` dataset.
4. Prepare the `cvevc_cve_commit_mappings` dataset by running `data_mappings.ipynb` in `notebooks/`. This dataset maps CVEs to commits.
5. Replicate `PatchFinder` by following instructions in `baselines/PatchFinder/README.md`.
6. Prepare the `cvevc_candidates` dataset by running `data_candidates.ipynb` in `notebooks/`. This dataset contains the candidates for each CVE (both PatchFinder and Random).
7. Run the main agent evaluation by following instructions in `evaluation/methods2test_runnable/README.md`.
8. Evaluate `PatchFinder` by following instructions in `baselines/PatchFinder/README.md`.
9. Replicate and evaluate `VulFixMiner` by following instructions in `baselines/VulFixMiner/README.md`.
10. Replicate and evaluate `LLM4VFD` by following instructions in `baselines/LLM4VFD/README.md`.
11. Replicate and evaluate `CommitShield` by following instructions in `baselines/CommitShield/README.md`.
12. Analyze the results using the `results.ipynb` notebook in `analysis/`.
13. Generate the tables by running `tables.ipynb` in `analysis/`.

Due to the variability of deep learning, we provide both the trained models and the generated results. The results are available in the data/ directory. Metadata and links to the trained models can be found at here. Datasets are available at: methods2test_small, methods2test_meta, methods2test_runnable.

## Datasets & Resources
To support reproducibility, all datasets, experiment code, and results are publicly available. The results are available in the data/ directory. The CVEVC datasets include candidates, CVEs, commits, and mappings between CVEs and commits, available at: [cvevc_candidates](https://huggingface.co/datasets/andstor/cvevc_candidates), [cvevc_cve](https://huggingface.co/datasets/andstor/cvevc_cve), [cvevc_commits](https://huggingface.co/datasets/andstor/cvevc_commits), and [cvevc_cve_commit_mappings](https://huggingface.co/datasets/andstor/cvevc_cve_commit_mappings). Experiment trajectories are available at [favia_trajectories](https://huggingface.co/datasets/andstor/favia_trajectories), and interactive traces can be explored through the demo spaces for [Top-10 PatchFinder](https://huggingface.co/spaces/andstor/phoenix-cvevc_candidates_PatchFinder_top10) and [Random 10](https://huggingface.co/spaces/andstor/phoenix-cvevc_candidates_random_10).

# XXXX 2025 replication package


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


git submodule update --init --recursive


### Phoenix Server

Run Phoenix server to collect traces:
```bash
PHOENIX_WORKING_DIR=phoenix python -m phoenix.server.main serve
```



## Requirements

### Dependencies
Please make sure you have Docker installed on your machine. See the [Docker installation guide](https://docs.docker.com/get-docker/) for more information.

Install the Python dependencies defined in the `requirements.txt`.
```bash
pip install -r requirements.txt
```


## Execution
The runnable methods2test codes are executed using the `evaluate_tests.py` script. This needs to be sandboxed due to potential security issues with executing arbitrary generated code. We use Docker for this purpose. Execute at your own risk.

### Build
Build the image (/evaluation/methods2test_runnable/Dockerfile) from the current directory:

```bash
docker build -t cve_agent .
```

### Usage

> [!CAUTION]
> Please execute following commands with caution! Generated codes might have unexpected behaviors. Execute at your own risk.

#### Validate Buildable Repos

Start a container using one of the following:

```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/tmp,target=/workspace/tmp \
  --mount type=bind,source="$(pwd)"/data/output/,target=/workspace/data/output \
  --mount type=bind,source="$(pwd)"/application.log/,target=/workspace/application.log \
  cve_agent python -u main.py --cve CVE-2019-6976
```


### Apptainer
If you want to use Apptainer instead of Docker, we provide pre-built images on GitHub Container Registry. The image is available at `ghcr.io/andstor/peft-unit-test-generation-replication-package/methods2test_runnable:main`.

Because we are executing untrusted code, we recommend using the `--containall` and `--no-home` flags to prevent the container from accessing your home directory and other sensitive files. This will require the use of an overlay file to store intermediate dependencies and results.

You can create an overlay file using the following command:

```bash
apptainer overlay create --size 10240 overlay.img
```

For more information on how to use Apptainer, please refer to the [Apptainer documentation](https://apptainer.org/docs/user/latest/).


```bash
apptainer run \
  --containall \
  --no-home \
  --overlay overlay.img \
  --cwd "/workspace/evaluation/methods2test_runnable/" \
  --mount type=bind,source="$(pwd)"/.tmp/,target=/workspace/evaluation/methods2test_runnable/tmp \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/coverage/,target=/workspace/data/methods2test_runnable/coverage \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/fixed/,target=/workspace/data/methods2test_runnable/fixed,readonly \
  docker://ghcr.io/andstor/peft-unit-test-generation-replication-package/methods2test_runnable:main python -u evaluate_tests.py --num_proc 20
```
    



    21c21e555dedfc2a6e7823625014235186047b6f

    https://github.com/locutusjs/locutus/commit/eb863321990e7e5514aa14f68b8d9978ece9e65e
    https://github.com/locutusjs/locutus/commit/21c21e555dedfc2a6e7823625014235186047b6f





    https://app.phoenix.arize.com/s/andstor/projects




Start the OpenTelemetry Collector with the file exporter and Phoenix exporter:

```bash
docker compose up
```

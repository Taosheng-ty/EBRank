# Empirical Bayesian for cold start problem in Online Learning to Rank 
## Create the Python environment.
First, Install the anaconda.
Then, create an virtual python environment,

    conda env create -f environment.yml
Activate the environment,
    conda activate EBRanker
## Generate the experiment setting,
please first run the following to generate experiment settings,

    python scripts/datascriptsEBLTR/generatingSetting.py

### Download the preprocessed LTR datasets from the following link and unzip it.

https://drive.google.com/file/d/17NDVk354G2Zv9_e2_63id_Ng0T-V9E6u/view?usp=sharing

After you unzipped it, you need to configure  LTRlocal_dataset_info.txt to specify where you put those datasets.
## Run the experiments,
With the setting.json generated from above cmd, you can run the following to run a batch of scripts.

    slurm_python --CODE_PATH=. --Cmd_file=main.py --JSON_PATH=localOutput/Jun15LTR/ --plain_script  --jobs_limit=10  --json2args  --white_list=PDGD+False

## Organize the results,
    python scripts/datascriptsEBLTR/resultTemporalPlot.py
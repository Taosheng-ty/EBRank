# Empirical Bayesian for cold start problem in Online Learning to Rank 
## Create the Python environment.
First, Install the anaconda.
Then, create an virtual python environment,

    conda env create -f environment.yml
Activate the environment,
    conda activate EBRanker


## Specify the data directory 
Please download datasets and  you need to configure  LTRlocal_dataset_info.txt to specify where you put those datasets.
## Run the experiments,
With the setting.json generated from above cmd, you can run the following to run a batch of scripts.

    python  main.py   --progressbar=false --rankListLength=5 --query_least_size=5 --positionBiasSeverity=1 --n_iteration=57505 --NewItemEnterProb=1.0 --dataset_name=MQ2007 --ExpandFeature=True --Ranker=NNTopK --random_seed=3 --log_dir=localOutput

You can choose different datasets, Ranke, ExpandFeature or not.

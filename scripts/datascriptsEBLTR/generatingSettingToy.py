import sys
import json
import os
from collections import defaultdict
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
import BatchExpLaunch.tools as tools
def write_setting(datasets,list_settings,settings_base,dataset_dict=None):
    """
    This function write settings.json which specify the parameters of main function.
    """
    
    for dataset in datasets:
        list_settings_data=dict(list_settings)
        list_settings_data["dataset_name"]=[dataset]
        list_settings_data = {k: list_settings_data[k] for k in desired_order_list if k in list_settings_data}
        setting_data=dict(settings_base)
        setting_data={**setting_data, **dataset_dict[dataset]} 
        print(root_path,"x"*100)
        tools.iterate_settings(list_settings_data,setting_data,path=root_path) 


settings_base={
        "progressbar":"false",
        "rankListLength":5,
        "query_least_size":5,
        "positionBiasSeverity":1
        # "NewItemEnterProb":0.1
        # "NumDocMaximum":20,
        # "relvance_strategy":"EstimatedAverage"
        }
# root_path="localOutput/Feb182022Data/"
positionBiasSeverity=[1]
root_path="localOutput/July26MQ2007Toy/"
desired_order_list=['NewItemEnterProb',"dataset_name","ExpandFeature","Ranker","sparseTraining","exploreParam","random_seed"]


##############################post-processing
datasets=["MQ2007"]
def change_datasetDict(NewItemEnterProb,dataset_dict):
        result=defaultdict(dict)
        for key in dataset_dict:
                result[key]["n_iteration"]=int(dataset_dict[key]["n_iteration"]/NewItemEnterProb)
        return result
        
dataset_dictBase={"MQ2008":{"n_iteration":728*20},"MQ2007":{"n_iteration":1643*35}}
NewItemEnterProbs=[0.1,0.4,0.7,1.0]
NewItemEnterProbs=[1.0]
# NewItemEnterProbs=[1.0]
for NewItemEnterProb in NewItemEnterProbs:
        NewDataset_dict=change_datasetDict(NewItemEnterProb,dataset_dictBase)
        print(NewDataset_dict)


        # list_settings={"ExpandFeature":['True',"False"],"Ranker":['TD_DBGD', 'PDGD',"TD_MGD", 'NNTopK',"NNRandomK","NNEpsilon"],\
        #         "NewItemEnterProb":[NewItemEnterProb],"random_seed":[0,1,2,3,4]}
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)

        # list_settings={"ExpandFeature":['True',"False"],"Ranker":["NNEpsilon"],"exploreParam":[1],\
        #         "NewItemEnterProb":[NewItemEnterProb],"random_seed":[0,1,2,3,4]}
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)
        # ##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
        # list_settings={'positionBiasSeverity':positionBiasSeverity,"ExpandFeature":['False'],"Ranker":["UCBRank","EBRankV1","EBRank"],"exploreParam":[0,1],\
        #         "NewItemEnterProb":[NewItemEnterProb],"random_seed":[0,1,2,3,4]}
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)  
        
        list_settings={'positionBiasSeverity':positionBiasSeverity,"ExpandFeature":['True'],"Ranker":["NNPriorFeature"],\
                "NewItemEnterProb":[NewItemEnterProb],"random_seed":[0,1,2,3,4]}
        write_setting(datasets,list_settings,settings_base,NewDataset_dict)         
              
        #write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
        # list_settings={"ExpandFeature":['True', 'False'],"Ranker":['TD_DBGD','PDGD',"TD_MGD", 'NNTopK',"NNRandomK"],\
        #         "NewItemEnterProb":[NewItemEnterProb],"random_seed":[0,1,2,3,4]}
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)
        
        # list_settings={"ExpandFeature":['False'],"Ranker":['BM25'],\
        #         "NewItemEnterProb":[NewItemEnterProb],"random_seed":[0,1,2,3,4]}
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)

        # ##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
        # list_settings={'positionBiasSeverity':positionBiasSeverity,"ExpandFeature":['False'],"Ranker":["UCBRank","EBRankOnlyBehaviour"],"exploreParam":[0,1,10],\
        #         "NewItemEnterProb":[NewItemEnterProb],"random_seed":[0,1,2,3,4]}
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)


        # ##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
        # list_settings={'positionBiasSeverity':positionBiasSeverity,"ExpandFeature":['False'],"Ranker":["EBRankOnlyPrior","EBRank","EBRankV1","EBRankSum"],"exploreParam":[0],\
        #         "NewItemEnterProb":[NewItemEnterProb], "random_seed":[0,1,2,3,4]}
        
        
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)
        # ##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
        # list_settings={'positionBiasSeverity':positionBiasSeverity,"ExpandFeature":['False'],"Ranker":["EBRankV1"],"exploreParam":[0,0.1,0.5,1.0,5.0,10.0,20.0,100.0,1000.0],\
        #         "NewItemEnterProb":[NewItemEnterProb], "random_seed":[0,1,2,3,4]}
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)

        # ##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
        # list_settings={'positionBiasSeverity':positionBiasSeverity,"ExpandFeature":['False'],"Ranker":["UCBRank"],"exploreParam":[0,0.005,0.01,0.05,0.1,0.2],\
        #         "NewItemEnterProb":[NewItemEnterProb], "random_seed":[0,1,2,3,4]}
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)

        # ##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
        # list_settings={'positionBiasSeverity':positionBiasSeverity,"ExpandFeature":['True', 'False'],"Ranker":["NNEpsilon"],"exploreParam":[0,1,10],\
        #         "NewItemEnterProb":[NewItemEnterProb],"random_seed":[0,1,2,3,4]}
        # write_setting(datasets,list_settings,settings_base,NewDataset_dict)






import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
from collections import OrderedDict
# sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import BatchExpLaunch.results_org as results_org
# import BatchExpLaunch.s as tools
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
def cal_timeFor1kLists(df):
    """
    This function return fairness for a dataframe
    """
    df["time1kLists"]=df["time"]/df["iterations"]*1000

step=19  
data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            # "MSLR-WEB10k":"MSLR10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"
            "MQ2008":"MQ2008",
            "MQ2007":"MQ2007",
            # "istella-s":"ist"
}
metric_name=[ "testWarmNDCG5", "testColdNDCG5",'test_NDCG_5_cumu','test_NDCG_5_aver']
# metric_name=['test_NDCG_1_cumu','test_NDCG_3_cumu','test_NDCG_5_cumu',"test_disparity","time1kLists"]
# metric_name=["test_disparity"]
metric_name_dict={"discounted_sum_test_ndcg":"Cum-NDCG","test_fairness":"bfairness","average_sum_test_ndcg":"average_cum_ndcg",\
    'f1_test_rel_fair':'crf-f1',"neg_test_exposure_disparity_not_divide_qfreq":"cnegdisparity",\
        'test_exposure_disparity_not_divide_qfreq':"Disparity","test_NDCG_5_cumu":"Cum-NDCG",\
            "testWarmNDCG5":"Warm-NDCG","testColdNDCG5":"Warm-NDCG","test_NDCG_5_aver":"a-NDCG",\
                "result_time_stamp_ctr_least_label1.0":"CTR"}
positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]
path_root="localOutput/Feb182022Data/"
path_root="localOutput/Feb192022DataTrueAver/"
path_root="localOutput/Feb192022DataEstimatedAverage/"
path_root="localOutput/Feb222022Data/relvance_strategy_EstimatedAverage"
path_root="localOutput/Feb222022Data/relvance_strategy_TrueAverage"
path_root="localOutput/Apr252022LTR_small/relvance_strategy_TrueAverage"
path_root="localOutput/Apr262022LTR/relvance_strategy_TrueAverage"
path_root="localOutput/Jun23LTR"
path_root="localOutput/July11LTR"
path_root="localOutput/July16MQ2007"
# path_root="localOutput/Apr30QPFairLTR/relvance_strategy_TrueAverage"

# path_root="localOutput/Apr252022LTR_more/relvance_strategy_EstimatedAverage"

# path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
NewItemEnterProbs=["NewItemEnterProb_0.1","NewItemEnterProb_0.3","NewItemEnterProb_0.5","NewItemEnterProb_0.7","NewItemEnterProb_1.0"]
NewItemEnterProbs=["NewItemEnterProb_0.4","NewItemEnterProb_0.7","NewItemEnterProb_1.0"]
NewItemEnterProbs=["NewItemEnterProb_1.0"]
ExpandFeatures=["ExpandFeature_False","ExpandFeature_True"]
yscale=results_org.setScaleFunction(a=1,b=1,low=False)
for NewItemEnterProb in NewItemEnterProbs:
    result_list=[]
    OutputPath=os.path.join(path_root,NewItemEnterProb,"result","Ablation")
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    for metrics in metric_name:
        for datasets,data_name_cur in data_rename.items():
            result_validated=OrderedDict()
            splits="dataset_name_"+datasets+"/"+"ExpandFeature_False"
            resultPath=os.path.join(path_root,NewItemEnterProb,splits)     
            if not os.path.isdir(resultPath):
                # print(path)       
                continue
            # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
            result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
            # result_validated["UCBw/oExp"]=result["Ranker_UCBRank"]["exploreParam_0"]
            result_validated["EBRank"]=result["Ranker_EBRankV1"]["exploreParam_20.0"]
            result_validated["W/o-Explo."]=result["Ranker_EBRankV1"]["exploreParam_0"]
            # result_validated["EBRank0.1"]=result["Ranker_EBRankV1"]["exploreParam_0.1"]
            # result_validated["EBRank1"]=result["Ranker_EBRankV1"]["exploreParam_1.0"]
            
            # result_validated["EBRank20"]=result["Ranker_EBRankV1"]["exploreParam_20.0"]
            # result_validated["EBRankw/Explore"]=result["Ranker_EBRank"]["exploreParam_1"]
            # result_validated["EBRank-w/o-Behav."]=result["Ranker_EBRankOnlyPrior"]["exploreParam_0"]
            # # result_validated["EBRankPriorw/Explore"]=result["Ranker_EBRankOnlyPrior"]["exploreParam_1"]
            # result_validated["EBRank-w/o-Prior"]=result["Ranker_EBRankOnlyBehaviour"]["exploreParam_0"]
            # # result_validated["Behaviourw/Explo"]=result["Ranker_EBRankOnlyBehaviour"]["exploreParam_1"]
            # result_validated["EBRankwExp"]=result["Ranker_EBRank"]["exploreParam_1"]
            result_validated["Only-Prior"]=result["Ranker_EBRankOnlyPrior"]["exploreParam_0"]
            # result_validated["EBRankPriorw/Explore"]=result["Ranker_EBRankOnlyPrior"]["exploreParam_1"]
            result_validated["Only-Behav."]=result["Ranker_EBRankOnlyBehaviour"]["exploreParam_0"]
            ########
            fig, ax = plt.subplots(figsize=(6.4,3.8))
            results_org.plot_metrics(result_validated,metrics,ax=ax)
            ax.set_yscale("function",functions=yscale)
            plt.locator_params(axis="y",nbins=5)
            for line in ax.lines:
                line.set_marker(None)
            plt.xlabel("Time steps")
            plt.ylabel(metric_name_dict[metrics])
            # plt.legend()
            ax.legend(bbox_to_anchor=(1.05, 1.05))  
            plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
            # plt.title(metrics+"---"+data_name_cur)
            plt.savefig(os.path.join(OutputPath,data_name_cur+metrics+"TimeAblation.pdf"), dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close()
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
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
metric_name=["test_NDCG_1_aver","test_SumClicks_cumu", "testWarmNDCG5", "testColdNDCG5"]
# metric_name=['test_NDCG_1_cumu','test_NDCG_3_cumu','test_NDCG_5_cumu',"test_disparity","time1kLists"]
# metric_name=["test_disparity"]
metric_name_dict={"discounted_sum_test_ndcg":"Cum-NDCG","test_fairness":"bfairness","average_sum_test_ndcg":"average_cum_ndcg",\
    'f1_test_rel_fair':'crf-f1',"neg_test_exposure_disparity_not_divide_qfreq":"cnegdisparity",\
        'test_exposure_disparity_not_divide_qfreq':"Disparity"}
positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]

NewItemEnterProbs=["NewItemEnterProb_0.1","NewItemEnterProb_0.3","NewItemEnterProb_0.5","NewItemEnterProb_0.7","NewItemEnterProb_1.0"]
path_root="localOutput/Feb182022Data/"
path_root="localOutput/Feb192022DataTrueAver/"
path_root="localOutput/Feb192022DataEstimatedAverage/"
path_root="localOutput/Feb222022Data/relvance_strategy_EstimatedAverage"
path_root="localOutput/Feb222022Data/relvance_strategy_TrueAverage"
path_root="localOutput/Apr252022LTR_small/relvance_strategy_TrueAverage"
path_root="localOutput/Apr262022LTR/relvance_strategy_TrueAverage"
path_root="localOutput/Jun23LTR"
path_root="localOutput/July11LTR"
# path_root="localOutput/Apr30QPFairLTR/relvance_strategy_TrueAverage"

# path_root="localOutput/Apr252022LTR_more/relvance_strategy_EstimatedAverage"

# path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
ExpandFeatures=["ExpandFeature_False","ExpandFeature_True"]
for NewItemEnterProb in NewItemEnterProbs:
    result_list=[]
    OutputPath=os.path.join(path_root,NewItemEnterProb,"result","Temporal")
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    for metrics in metric_name:
        for datasets,data_name_cur in data_rename.items():
            for ExpandFeature in ExpandFeatures:
                result_validated={}
                splits="dataset_name_"+datasets+"/"+ExpandFeature
                resultPath=os.path.join(path_root,NewItemEnterProb,splits)
                if not os.path.isdir(resultPath):
                    # print(path)       
                    continue
                # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
                result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
                results_org.iterate_applyfunction(result,cal_timeFor1kLists)
                result_validated["DBGD"]=result["Ranker_TD_DBGD"]
                result_validated["MGD"]=result["Ranker_TD_MGD"]
                result_validated["PDGD"]=result["Ranker_PDGD"]
                result_validated["NNRandomK"]=result["Ranker_NNRandomK"]
                result_validated["NNTopK"]=result["Ranker_NNTopK"]
                results_org.plot_metrics(result_validated,metrics)
                plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
                plt.title(metrics+"---"+data_name_cur)
                plt.savefig(os.path.join(OutputPath,ExpandFeature+data_name_cur+metrics+"Time.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
                plt.close()
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
            "MSLR-WEB30k_beh_rm":"MSLR-WEB30k",\
            "MSLR-WEB10k_beh_rm":"MSLR10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"
            # "MQ2008":"MQ2008",
            "MQ2007":"MQ2007",
            # "istella-s":"ist"
}
metric_name=[ "result_time_stamp_ctr_least_label1.0","testWarmNDCG5", "testColdNDCG5",'test_NDCG_5_cumu','test_NDCG_5_aver']
# metric_name=[ "result_time_stamp_ctr_least_label1.0"]
# metric_name=['test_NDCG_1_cumu','test_NDCG_3_cumu','test_NDCG_5_cumu',"test_disparity","time1kLists"]
# metric_name=["test_disparity"]
metric_name_dict={"discounted_sum_test_ndcg":"Cum-NDCG","test_fairness":"bfairness","average_sum_test_ndcg":"average_cum_ndcg",\
    'f1_test_rel_fair':'crf-f1',"neg_test_exposure_disparity_not_divide_qfreq":"cnegdisparity",\
        'test_exposure_disparity_not_divide_qfreq':"Disparity","test_NDCG_5_cumu":"c-NDCG@5",\
            "testWarmNDCG5":"NDCG@5 (Warm)","testColdNDCG5":"NDCG@5 (Cold)","test_NDCG_5_aver":"a-NDCG@5",\
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
path_root="localOutput/July14StartLTRv2"
path_root="localOutput/July16MQ2007"
# path_root="localOutput/July20MSLR"
# path_root="localOutput/July26MQ2007Toy"
# path_root="localOutput/Apr30QPFairLTR/relvance_strategy_TrueAverage"

# path_root="localOutput/Apr252022LTR_more/relvance_strategy_EstimatedAverage"

# path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
NewItemEnterProbs=["NewItemEnterProb_0.1","NewItemEnterProb_0.3","NewItemEnterProb_0.5","NewItemEnterProb_0.7","NewItemEnterProb_1.0"]
NewItemEnterProbs=["NewItemEnterProb_0.1","NewItemEnterProb_0.5","NewItemEnterProb_0.7","NewItemEnterProb_1.0"]
NewItemEnterProbs=["NewItemEnterProb_1.0"]
# NewItemEnterProbs=["NewItemEnterProb_0.1"]
ExpandFeatures=["ExpandFeature_False","ExpandFeature_True"]
for NewItemEnterProb in NewItemEnterProbs:
    result_list=[]
    OutputPath=os.path.join(path_root,NewItemEnterProb,"result","Temporal")
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    
    for datasets,data_name_cur in data_rename.items():
        result_validated={}
        splits="dataset_name_"+datasets+"/"+"ExpandFeature_True"
        resultPath=os.path.join(path_root,NewItemEnterProb,splits)
        if not os.path.isdir(resultPath):
            print(resultPath)       
            continue
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        results_org.iterate_applyfunction(result,cal_timeFor1kLists)
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        result_validated["DBGD"]=result["Ranker_TD_DBGD"]
        result_validated["MGD"]=result["Ranker_TD_MGD"]
        result_validated["PDGD"]=result["Ranker_PDGD"]
        # result_validated["PDGDv1"]=result["Ranker_PDGDv1"]
        result_validated["NNRandomK"]=result["Ranker_NNRandomK"]
        result_validated["NNTopK"]=result["Ranker_NNTopK"]
        result_validated["NNEpsilon"]=result["Ranker_NNEpsilon"]["exploreParam_1"]
        # result_validated["NNEpsilon"]=result["Ranker_NNEpsilon"]["exploreParam_0"]
        #######
        splits="dataset_name_"+datasets+"/"+"ExpandFeature_False"  
        resultPath=os.path.join(path_root,NewItemEnterProb,splits) 
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)

        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        # result_validated["UCBw/oExp"]=result["Ranker_UCBRank"]["exploreParam_0"]
        # result_validated["NNTopKw/oFeature"]=result["Ranker_NNTopK"]
        # result_validated["DBGDWO"]=result["Ranker_TD_DBGD"]
        # result_validated["MGDWO"]=result["Ranker_TD_MGD"]
        # result_validated["PDGDWO"]=result["Ranker_PDGD"]
        # result_validated["UCBRank_1"]=result["Ranker_UCBRank"]["exploreParam_1"]
        result_validated["UCBRank"]=result["Ranker_UCBRank"]["exploreParam_0"]
        result_validated["UCBRank0.005"]=result["Ranker_UCBRank"]["exploreParam_0.005"]
        # result_validated["BM25"]=result["Ranker_BM25"]
        # result_validated["EBRank"]=result["Ranker_EBRank"]["exploreParam_0"]
        result_validated["EBRank"]=result["Ranker_EBRankV1"]["exploreParam_0"]
        
        # result_validated["EBRankSum"]=result["Ranker_EBRankSum"]["exploreParam_0"]
        # result_validated["EBRankwExp"]=result["Ranker_EBRank"]["exploreParam_1"]
        ########
        for metrics in metric_name:
            plt.xlabel("Time steps")
            plt.ylabel(metric_name_dict[metrics])
            results_org.plot_metrics(result_validated,metrics)
            plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
            plt.legend(bbox_to_anchor=(1.1, 1.05))  
            # plt.title(metrics+"---"+data_name_cur)
            plt.savefig(os.path.join(OutputPath,data_name_cur+metrics+"Time.pdf"), dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close()
import sys
import os
import pandas as pd

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
            # # "News":"News",\
            "MSLR-WEB30k_beh_rm":"MSLR-WEB30k",\
            "MSLR-WEB10k_beh_rm":"MSLR10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s",
            # "MQ2008":"MQ2008",
            # "MQ2007":"MQ2007",
}
# metric_name=['test_NDCG_1_aver','test_NDCG_3_aver','test_NDCG_5_aver',"test_disparity","time1kLists"]
# metric_name=['test_NDCG_5_cumu',"testColdNDCG5","test_ClickEntrpy","test_SumClicks_cumu"]
metric_name=["test_NDCG_5_cumu","testWarmNDCG5","testColdNDCG5"]
metric_name_dict={"discounted_sum_test_ndcg":"Cum-NDCG","test_fairness":"bfairness","average_sum_test_ndcg":"average_cum_ndcg",\
    'f1_test_rel_fair':'crf-f1',"neg_test_exposure_disparity_not_divide_qfreq":"cnegdisparity",\
        'test_exposure_disparity_not_divide_qfreq':"Disparity"}
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
path_root="localOutput/Apr252022LTR/relvance_strategy_TrueAverage"
path_root="localOutput/Apr262022LTR/relvance_strategy_EstimatedAverage"
path_root="localOutput/July16MQ2007"
path_root="localOutput/July20MSLR"
# path_root="localOutput/July26MQ2007Toy"
NewItemEnterProbs=["NewItemEnterProb_0.1","NewItemEnterProb_0.5","NewItemEnterProb_0.7","NewItemEnterProb_1.0"]
NewItemEnterProbs=["NewItemEnterProb_1.0"]
# path_root="localOutput/Apr30QPFairLTR/relvance_strategy_EstimatedAverage"
ExpandFeatures=["ExpandFeature_False","ExpandFeature_True"]
# path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
for NewItemEnterProb in NewItemEnterProbs:
    result_list=[]
    OutputPath=os.path.join(path_root,NewItemEnterProb,"result","ResultTable")
    
    for datasets,data_name_cur in data_rename.items():
        result_validated={}
        splits="dataset_name_"+datasets+"/"+ExpandFeatures[0]
        resultPath=os.path.join(path_root,NewItemEnterProb,splits)
        
        if not os.path.isdir(resultPath):
            # print(path)       
            continue
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        results_org.iterate_applyfunction(result,cal_timeFor1kLists)
        result_validated["BM25"]=result["Ranker_BM25"]
        result_validated["DBGD"]=result["Ranker_TD_DBGD"]
        result_validated["MGD"]=result["Ranker_TD_MGD"]
        result_validated["PDGD"]=result["Ranker_PDGD"]
        result_validated["CFTopK"]=result["Ranker_NNTopK"]
        result_validated["CFRandomK"]=result["Ranker_NNRandomK"]
        # result_validated["CFEpsilon"]=result["Ranker_NNEpsilon"]["exploreParam_1"]
        # result_validated["EBRankPrior"]=result["Ranker_EBRankOnlyPrior"]["exploreParam_0"]
        result_validated["EBRank"]=result["Ranker_EBRank"]["exploreParam_0"]
        result_validated["EBRankV1"]=result["Ranker_EBRankV1"]["exploreParam_0"]
        
        # result_validated["TS"]=result["Ranker_TS"]
        # result_validated["EBRankV1_1"]=result["Ranker_EBRankV1"]["exploreParam_1.0"]
        # result_validated["EBRankV1_0.1"]=result["Ranker_EBRankV1"]["exploreParam_0.1"]
        # result_validated["UCBRank_0.1"]=result["Ranker_UCBRank"]["exploreParam_0.1"] 
        # result_validated["UCBRank_0.005"]=result["Ranker_UCBRank"]["exploreParam_0.005"]  
        result_validated["UCBRank_0"]=result["Ranker_UCBRank"]["exploreParam_0"]              
        splits="dataset_name_"+datasets+"/"+ExpandFeatures[1]
        resultPath=os.path.join(path_root,NewItemEnterProb,splits)            
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        result_validated["ZDBGD"]=result["Ranker_TD_DBGD"]
        result_validated["ZMGD"]=result["Ranker_TD_MGD"]
        result_validated["ZPDGD"]=result["Ranker_PDGD"]
        result_validated["ZCFTopK"]=result["Ranker_NNTopK"]
        result_validated["ZCFRandomK"]=result["Ranker_NNRandomK"]
        result_validated["Prior"]=result["Ranker_NNPriorFeature"]
        # result_validated["ZCFEpsilon"]=result["Ranker_NNEpsilon"]["exploreParam_1"]
        results_org.iterate_applyfunction(result,cal_timeFor1kLists)
        for metrics in metric_name:
            result_vali_metrics=results_org.extract_step_metric(result_validated,metrics,step,data_name_cur+metrics)
            result_list=result_list+result_vali_metrics

    result_list=results_org.filteroutNone(result_list)
    result_dfram=pd.DataFrame(result_list, columns=["method","datasets","metrics"])
    result_dfram=result_dfram.pivot(index='method', columns='datasets', values='metrics')
    r,rstd=results_org.to_latex(result_dfram)
    os.makedirs(OutputPath, exist_ok=True)
    output_path=os.path.join(OutputPath,NewItemEnterProb+"mean_latex.csv")
    r.to_csv(output_path)
    output_path=os.path.join(OutputPath,NewItemEnterProb+"mean_std_latex.csv")
    rstd.to_csv(output_path)
    mean=results_org.to_mean(result_dfram)
    output_path=os.path.join(OutputPath,NewItemEnterProb+"mean.csv")
    mean.to_csv(output_path)
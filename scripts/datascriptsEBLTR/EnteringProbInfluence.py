import sys
import os
import pandas as pd
import config
import numpy as np
import BatchExpLaunch.results_org as results_org
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams['pdf.fonttype']=42
font = {'size'   : 14}

matplotlib.rc('font', **font)

from collections import OrderedDict
# import BatchExpLaunch.s as tools
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
def cal_timeFor1kLists(df):
    """
    This function return fairness for a dataframe
    """
    df["time1kLists"]=df["time"]/df["iterations"]*1000

step=19  

#Specify the datasets 

data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            # "MSLR-WEB10k":"MSLR10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s",
            # "MQ2008":"MQ2008",
            "MQ2007":"MQ2007",
}
# metric_name=['test_NDCG_1_aver','test_NDCG_3_aver','test_NDCG_5_aver',"test_disparity","time1kLists"]
# metric_name=['test_NDCG_5_cumu',"testColdNDCG5","test_ClickEntrpy","test_SumClicks_cumu"]
#Specify the metrics we want to investigate
metric_name=["test_NDCG_5_aver","test_NDCG_5_cumu","testWarmNDCG5","testColdNDCG5"]
metric_name_dict={"discounted_sum_test_ndcg":"Cum-NDCG","test_fairness":"bfairness","average_sum_test_ndcg":"average_cum_ndcg",\
    'f1_test_rel_fair':'crf-f1',"neg_test_exposure_disparity_not_divide_qfreq":"cnegdisparity",\
        'test_exposure_disparity_not_divide_qfreq':"Disparity","test_NDCG_5_cumu":"Cum-NDCG",\
            "testWarmNDCG5":"Cold-NDCG","testColdNDCG5":"Warm-NDCG","test_NDCG_5_aver":"a-NDCG",\
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
path_root="localOutput/Apr252022LTR/relvance_strategy_TrueAverage"
path_root="localOutput/Apr262022LTR/relvance_strategy_EstimatedAverage"
#Specify data results folder
path_root="localOutput/July16MQ2007"
# path_root="localOutput/July20MQ2007"
NewItemEnterProbs=["NewItemEnterProb_0.1","NewItemEnterProb_0.4","NewItemEnterProb_0.7","NewItemEnterProb_1.0"]
# path_root="localOutput/Apr30QPFairLTR/relvance_strategy_EstimatedAverage"
ExpandFeatures=["ExpandFeature_False","ExpandFeature_True"]
# path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
result_list=[]
for NewItemEnterProb in NewItemEnterProbs:
    
    OutputPath=os.path.join(path_root,NewItemEnterProb,"result","ResultTable")
    
    for datasets,data_name_cur in data_rename.items():
        result_validated=OrderedDict()

        splits="dataset_name_"+datasets+"/"+ExpandFeatures[1]
        resultPath=os.path.join(path_root,NewItemEnterProb,splits)            
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        result_validated["DBGD"]=result["Ranker_TD_DBGD"]
        result_validated["MGD"]=result["Ranker_TD_MGD"]
        result_validated["PDGD"]=result["Ranker_PDGD"]
        result_validated["CFTopK"]=result["Ranker_NNTopK"]
        result_validated["CFRandomK"]=result["Ranker_NNRandomK"]
        result_validated["CFEpsilon"]=result["Ranker_NNEpsilon"]["exploreParam_1"]
        result_validated["PRIOR"]=result["Ranker_NNPriorFeature"]

        splits="dataset_name_"+datasets+"/"+ExpandFeatures[0]
        resultPath=os.path.join(path_root,NewItemEnterProb,splits)
        
        if not os.path.isdir(resultPath):
            # print(path)       
            continue
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        results_org.iterate_applyfunction(result,cal_timeFor1kLists)
        # result_validated["BM25"]=result["Ranker_BM25"]
        # result_validated["DBGD"]=result["Ranker_TD_DBGD"]
        # result_validated["MGD"]=result["Ranker_TD_MGD"]
        # result_validated["PDGD"]=result["Ranker_PDGD"]
        # result_validated["CFTopK"]=result["Ranker_NNTopK"]
        # result_validated["CFRandomK"]=result["Ranker_NNRandomK"]
        # result_validated["CFEpsilon"]=result["Ranker_NNEpsilon"]["exploreParam_1"]
        # result_validated["EBRank(Ours)"]=result["Ranker_EBRank"]["exploreParam_0"]
        result_validated["EBRank(Ours)"]=result["Ranker_EBRankV1"]["exploreParam_0"]
        # result_validated["UCBRank_1"]=result["Ranker_UCBRank"]["exploreParam_1"]   
        result_validated["UCBRank"]=result["Ranker_UCBRank"]["exploreParam_0"]      
        result_validated["BM25"]=result["Ranker_BM25"]
        # results_org.iterate_applyfunction(result,cal_timeFor1kLists)
        for metrics in metric_name:
            result_vali_metrics=results_org.extract_step_metric(result_validated,metrics,step,NewItemEnterProb+data_name_cur+metrics)
            result_list=result_list+result_vali_metrics

result_list=results_org.filteroutNone(result_list)
result_dfram=pd.DataFrame(result_list, columns=["method","datasets","metrics"])
result_dfram=result_dfram.pivot(index='method', columns='datasets', values='metrics')
# mean=result_dfram.applymap(func=np.mean)
mean=result_dfram
os.makedirs(OutputPath, exist_ok=True)
output_path=os.path.join(path_root,"result","mean_latex.csv")
OutputPath=os.path.join(path_root,"result","ResultTable")
os.makedirs(OutputPath, exist_ok=True)
mean.to_csv(output_path)
x=[0.1,0.4,0.7,1.0]
# desiredGradFairColorDict={"EBRank(Ours)":"b"}
yMQfunctions=results_org.setScaleFunction(a=190,b=1,low=False)
for datasets,data_name_cur in data_rename.items():
    for metrics in metric_name:
        fig, ax = plt.subplots(figsize=(6.4,3.8))
        resultKey=[]
        for NewItemEnterProb in NewItemEnterProbs:
            resultKey.append(NewItemEnterProb+data_name_cur+metrics)
        CurDataMetricResult=mean[resultKey]
        result_dict=OrderedDict()
        for index in result_validated.keys():
            result_dict[index]=[x,CurDataMetricResult.loc[index].to_list()]
        results_org.plot(result_dict,ax=ax,desiredColorDict=config.desiredColor,desiredMarkerDict=config.desiredMarker,errbar=True)
        # for line in ax.lines:
        #     line.set_marker(None)
        ax.set_yscale("function",functions=yMQfunctions)
        ax.set_yticks(ticks=[80,120,160,180])
        ax.set_xticks(x)
        ax.set_xlabel("New item entering probability $\eta$ .")
        ax.set_ylabel(metric_name_dict[metrics])
        ax.legend(bbox_to_anchor=(1.05, 1.05))   
        # ax.legend(ncol=4,loc='best', fancybox=True, framealpha=0.5)
        
        output_path=os.path.join(path_root,"result",datasets+metrics+"EnteringProb.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close(fig)

        
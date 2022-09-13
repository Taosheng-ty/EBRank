import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
# sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import BatchExpLaunch.results_org as results_org
import numpy as np
import itertools
# import BatchExpLaunch.s as tools
def smooth(y, box_pts=10):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
def plot_metrics(name_results_pair:dict,plots_y_partition:str="metrics_NDCG",errbar=True,
plots_x_partition:str="iterations",groupby="iterations",ax=None,graph_param=None,smoooth_fn=None)->None:
    
    '''    
        name_results_pair:{method_name:result_dataframe}
        plots_partition: key name in each result_dataframe which need to be plotted
    '''
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_list = prop_cycle.by_key()['color']
    colors=itertools.cycle(colors_list)
    marker = itertools.cycle((',', '+', '.', 'o', '*')) 
    if ax:
        plot=ax
    else:
        plot=plt
    for algo_name in name_results_pair:
            algo_result=name_results_pair[algo_name]
#             print(type(algo_result),"*"*100)
            mean_orig=algo_result.groupby(groupby).mean().reset_index()
           
            std=algo_result.groupby(groupby).std().reset_index()
            mean = mean_orig[mean_orig[plots_y_partition].notna()]
            if smoooth_fn is not None:
                mean[plots_y_partition]=smoooth_fn(mean[plots_y_partition])
                # errbar=False
            
            std = std[mean_orig[plots_y_partition].notna()]
            if plots_x_partition not in mean.keys() or plots_y_partition not in mean.keys() :
                continue
#             assert plots_y_partition in algo_result, algo_name+" doesn't contain the partition "+plots_y_partition
            if not errbar:
                plot.plot(mean[plots_x_partition],mean[plots_y_partition], marker = next(marker),color=next(colors), label=algo_name)
            else:
                plot.errorbar(mean[plots_x_partition],mean[plots_y_partition], yerr=std[plots_y_partition], marker = next(marker),color=next(colors), label=algo_name)
    # if ax is None:
    #     gca=plot.gca()
    #     gca.set(**graph_param)
    #     plot.legend()
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
            # "MQ2008":"MQ2008",
            "MQ2007":"MQ2007",
            # "istella-s":"ist"
}
metric_name=[ "result_time_stamp_ctr_least_label1.0","testWarmNDCG5", "testColdNDCG5",'test_NDCG_5_cumu','test_NDCG_5_aver']
metric_name=[ "result_time_stamp_ctr_least_label1.0"]
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
path_root="localOutput/July26MQ2007Toy"
# path_root="localOutput/July16MQ2007"
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
        # result_validated["NNEpsilon"]=result["Ranker_NNEpsilon"]["exploreParam_1"]
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
        # result_validated["UCBRank1"]=result["Ranker_UCBRank"]["exploreParam_1"]
        # result_validated["BM25"]=result["Ranker_BM25"]
        # result_validated["EBRank"]=result["Ranker_EBRank"]["exploreParam_0"]
        result_validated["EBRank"]=result["Ranker_EBRankV1"]["exploreParam_0"]
        # result_validated["EBRank1"]=result["Ranker_EBRankV1"]["exploreParam_1"]
        # result_validated["EBRankSum"]=result["Ranker_EBRankSum"]["exploreParam_0"]
        # result_validated["EBRankwExp"]=result["Ranker_EBRank"]["exploreParam_1"]
        ########
        for metrics in metric_name:
            fig, axs = plt.subplots(figsize=(6.4,2.4))
            axs.set_xlabel("Time step at which the item was introduced to the system")
            axs.set_ylabel("Click Rates")
            # results_org.plot_metrics(result_validated,metrics)
            plot_metrics(result_validated,metrics,ax=axs,smoooth_fn=smooth)
            for line in axs.lines:
                line.set_marker(None)
            plt.legend(bbox_to_anchor=(1.1, 1.05)) 
            plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
            # plt.title(metrics+"---"+data_name_cur)
            plt.savefig(os.path.join(OutputPath,data_name_cur+metrics+"Time.pdf"), dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close()
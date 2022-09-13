import numpy as np
from scipy.stats import entropy
import utils.rankings as rnk
def DCG(sampled_rankings, q_doc_weights,rank_weights):
    """
    This funciton return the DCG.
    """
    cutoff = sampled_rankings.shape[1]
    return np.sum(
                q_doc_weights[sampled_rankings]*rank_weights[None, :cutoff],
            axis=1)

def NDCG_based_on_samples(sampled_rankings,q_doc_weights,rank_weights,cutoff):
    """
    This funciton return the NDCG based multiple samples for the same query.
    """
    n_samples=sampled_rankings.shape[0]
    ranklistLength=sampled_rankings.shape[1]
    assert ranklistLength>=cutoff,"The length of a rank list should be greater than the NDCG cutoff"
    if q_doc_weights.sum()<=0:
      return np.zeros(n_samples)
    ideal_ranking=np.argsort(-q_doc_weights)[:cutoff][None,:]
    dcg=DCG(sampled_rankings[:,:cutoff],
        q_doc_weights,rank_weights)
    idcg=DCG(ideal_ranking,
        q_doc_weights,rank_weights)
    return dcg/idcg
def NDCG_based_on_single_sample(sampled_ranking,q_doc_weights,rank_weights,cutoff):
    """
    This funciton return the NDCG based a sinlge samples for a query.
    """
    sampled_ranking=sampled_ranking[None,:]
    NDCG=NDCG_based_on_samples(sampled_ranking,q_doc_weights,rank_weights,cutoff)[0]
    return NDCG


def UpdateOnlineMultipleCutoffsDatasplit(clicks,sampled_ranking,qid,dataSplit,rank_weights,cutoffs,NDCGDict):
    """
    First we return the NDCG @ different cutoffs based qid in a datasplit.
    """
    q_label_vector=dataSplit.query_values_from_vector(qid,dataSplit.label_vector,cold=False)
    for cutoff in cutoffs:
        dataSplitName=dataSplit.name
        nameCur="_".join([dataSplitName,"NDCG",str(cutoff)])
        NDCGCur=NDCG_based_on_single_sample(sampled_ranking,q_label_vector,rank_weights,cutoff)
        NDCGDict[nameCur].append(NDCGCur)
    WeightedClick=clicks *rank_weights
    for cutoff in cutoffs:
        dataSplitName=dataSplit.name
        nameCur="_".join([dataSplitName,"WeightedSumClick",str(cutoff)])
        WeightedClickCur=WeightedClick[:cutoff].sum()
        NDCGDict[nameCur].append(WeightedClickCur)
    nameCur="_".join([dataSplitName,"SumClicks"])
    NDCGDict[nameCur].append(clicks.sum())
    
        
               
def Update_NDCG_multipleCutoffsData(sampled_ranking,q_label_vector,rank_weights,cutoffs,NDCGDict):
    """
    This funciton return the NDCG @ different cutoffs.
    """
    for cutoff in cutoffs:
        nameCur="_".join(["NDCG",str(cutoff)])
        NDCGCur=NDCG_based_on_single_sample(sampled_ranking,q_label_vector,rank_weights,cutoff)
        NDCGDict[nameCur].append(NDCGCur)

def dicounted_metrics(metrics,gamma=0.995):
    """
    This funciton returns the discounted cumulative sum.
    """  
    m=len(metrics)
    results=np.zeros(m)
    previous_sum=0
    for i in range(m):
        previous_sum=previous_sum*gamma+metrics[i]
        results[i]=previous_sum
    return results
def outputCumulative(NDCGDict,OutputDict):
    """
    This funciton orgainze NDCG results.
    """    
    for key,value in NDCGDict.items():
        OutputDict[key+"_cumu"].append(dicounted_metrics(value,gamma=0.995)[-1])
        OutputDict[key+"_aver"].append(np.mean(value))
        OutputDict[key+"_cumuNoDiscount"].append(dicounted_metrics(value,gamma=1.0)[-1])

def disparity(exposure,rel,**kwargs):
    """
    This is an implementation of exposure disparity of a single query defined in Eq.29&30 in the following paper, 
    """    
    q_n_docs = rel.shape[0]
    swap_reward = exposure[:,None]*rel[None,:]
    q_result = np.sum((swap_reward-swap_reward.T)**2.)/(q_n_docs*(q_n_docs-1))
    return q_result
def disparityDivideFreq(exposure,rel,q_freq,**kwargs):
    """
    This is an implementation of exposure disparity of a single query defined in Eq.29&30 in the following paper, 
    """    
    q_result = disparity(exposure/q_freq,rel)
    return q_result
def L1(exposure,rel,**kwargs):
    """
    This function gives the l1 distance between two distribution.
    """  
    exposure=exposure/exposure.sum()
    rel=rel/rel.sum()
    return np.sum(np.abs(exposure-rel))/2
def evaluate_unfairness(data_split,fcn,effect="exposure"):
    """
    This is an implementation of exposure disparity of multiple queries defined in Eq.29&30 in the following paper, 
    Computationally Efficient Optimization ofPlackett-Luce Ranking Models for Relevance and Fairness. Harrie Oosterhuis SIGIR 2021 
    """
    queriesList=data_split.queriesList
    unfairness_list=[]
    for qid in queriesList:
        if effect=="exposure":
            q_eff=data_split.query_values_from_vector(qid,data_split.exposure)
        elif effect=="click":
            q_eff=data_split.query_values_from_vector(qid,data_split.ClickSum)
        else:
            raise
        q_rel=data_split.query_values_from_vector(qid,data_split.label_vector)
        q_freq=data_split.query_freq[qid]
        if q_freq<=0 or q_eff.sum()<=0 or q_rel.sum()<=0:
            continue
        unfairness_list.append(fcn(q_eff,q_rel,q_freq=q_freq))
    unfairness_list=np.array(unfairness_list)
    return np.mean(unfairness_list)
def evaluate_clickEntropy(data_split):
    """
    This is an implementation of exposure disparity of multiple queries defined in Eq.29&30 in the following paper, 
    Computationally Efficient Optimization ofPlackett-Luce Ranking Models for Relevance and Fairness. Harrie Oosterhuis SIGIR 2021 
    """
    queriesList=data_split.queriesList
    unfairness_list=[]
    for qid in queriesList:
        q_eff=data_split.query_values_from_vector(qid,data_split.ClickSum)
        q_eff=np.clip(q_eff,1e-5,np.inf)
        q_entropy=entropy(q_eff, base=2)
        unfairness_list.append(q_entropy)
    unfairness_list=np.array(unfairness_list)
    return np.mean(unfairness_list)
def evaluate_offlineNDCGDatasplit(ranker,data,rank_weights,cutoffs,OutputDict,Cold=True):
    """
    This is an implementation of exposure disparity of multiple queries defined in Eq.29&30 in the following paper, 
    Computationally Efficient Optimization ofPlackett-Luce Ranking Models for Relevance and Fairness. Harrie Oosterhuis SIGIR 2021 
    """
    dataSplits=[data.train,data.validation,data.test]
    NameColdWarm="Cold" if Cold else "Warm"
    for data_split in dataSplits:
        ndcgList=[]
        queriesList=data_split.queriesList
        SplitName=data_split.name
        for qid in queriesList:
            ranking=ranker.get_query_dataSplit_TestRanking(qid,data_split,Cold=Cold)
            q_label_vector=data_split.query_values_from_vector(qid,data_split.label_vector,cold=False)
            q_ndcgList=[]
            for cutoff in cutoffs:
                NDCGCur=NDCG_based_on_single_sample(ranking,q_label_vector,rank_weights,cutoff)
                q_ndcgList.append(NDCGCur)
            ndcgList.append(q_ndcgList)
        ndcgList=np.array(ndcgList)
        for ind,cutoff in enumerate(cutoffs):
            OutputDict[SplitName+NameColdWarm+"NDCG"+str(cutoff)].append(ndcgList[:,ind].mean())
def outputFairnessDatasplit(data,OutputDict):
    """
    This funciton orgainze fairness results.
    """   
    UnfairnessFcns={"disparity":disparity,"disparityDivideFreq":disparityDivideFreq,"L1":L1}
    dataSplits=[data.train,data.validation,data.test]
    for dataSplit in dataSplits:
        for fcnName,fcn in UnfairnessFcns.items():
            unfairness=evaluate_unfairness(dataSplit,fcn,effect="exposure")  ## get unfairness for exposure
            logName="_".join([dataSplit.name,fcnName,"exposure"])
            OutputDict[logName].append(unfairness)
            unfairness=evaluate_unfairness(dataSplit,fcn,effect="click")    ## get unfairness for click
            logName="_".join([dataSplit.name,fcnName,"click"])
            OutputDict[logName].append(unfairness)
        ClickEntrpy=evaluate_clickEntropy(dataSplit)
        logName="_".join([dataSplit.name,"ClickEntrpy"])
        OutputDict[logName].append(ClickEntrpy)
def LogStatisticsDatasplit(data,OutputDict):
    """
    This funciton orgainze fairness results.
    """   
    UnfairnessFcns={"disparity":disparity,"disparityDivideFreq":disparityDivideFreq,"L1":L1}
    dataSplits=[data.train,data.validation,data.test]
    for dataSplit in dataSplits:
        
        logName="_".join([dataSplit.name,"NumClickedItem"])
        NumClickedItem=int((dataSplit.ClickSum>0).sum())
        OutputDict[logName].append(NumClickedItem)
        logName="_".join([dataSplit.name,"NumImpressedItem"])
        NumImpressedItem=int((dataSplit.docFreq>0).sum())
        OutputDict[logName].append(NumImpressedItem)
        logName="_".join([dataSplit.name,"NumTotalAvailItem"])
        NumTotalAvailItem=int((dataSplit.AvaildocRange).sum())
        OutputDict[logName].append(NumTotalAvailItem)
        logName="_".join([dataSplit.name,"NumMaskedItem"])
        NumMaskedItem=int(dataSplit.feature_matrix.shape[0]-(dataSplit.AvaildocRange).sum())
        OutputDict[logName].append(NumMaskedItem)
def outputFairnessData(data,OutputDict):
    """
    This funciton orgainze fairness results.
    """   
    UnfairnessFcns={"disparity":disparity,"disparityDivideFreq":disparityDivideFreq,"L1":L1}
    for fcnName,fcn in UnfairnessFcns.items():
        q_exposure=data.exposure
        q_rel=data.TrueAverRating
        q_freq=data.queryFreq
        unfairness=fcn(q_exposure,q_rel,q_freq=q_freq)
        logName=fcnName
        OutputDict[logName].append(unfairness)
        
def time_stamp_clicks(clicks,label,shown_id_lists,select_label):
    """_summary_

    Args:
        clicks (_type_): _description_
        label (_type_): _description_
        shown_id_lists (_type_): _description_
        select_label (_type_): _description_

    Returns:
        _type_: _description_
    """
    result=[]
    for shown_id_list in shown_id_lists:
        if  shown_id_list is None:
            result.append(None)
            continue
        selected_id=shown_id_list[label[shown_id_list]>=select_label]
        clicks_cur=clicks[selected_id]
        result.append(clicks_cur.mean())
    return result

def SplitItemsBins(TimeStampEntering,evalIterations):
    """_summary_    Split items into bins according to their entering time.

    Args:
        TimeStampEntering (_type_): _description_
        evalIterations (_type_): _description_

    Returns:
        _type_: _description_
    """
    resultIdBin=[]
    for index,iteration in enumerate(evalIterations[:-1]):
        indBin=np.where((TimeStampEntering>=evalIterations[index]) * (TimeStampEntering<evalIterations[index+1]))[0]
        resultIdBin.append(indBin)
    return resultIdBin
    

def time_stamp_clicksDataSplit(DataSplit,least_label,evalIterations):
    """_summary_  

    Args:
        DataSplit (_type_): _description_
        least_label (_type_): _description_
        evalIterations (_type_): _description_
    """
    shown_id_lists=SplitItemsBins(DataSplit.TimeStampEntering,evalIterations)
    CTR=DataSplit.CalCTROverExistingTime(evalIterations[-1])
    label=DataSplit.label_vector
    resultCTR=time_stamp_clicks(CTR,label,shown_id_lists,least_label)
    return resultCTR

def ModelWeightEval(ranker,OutputDict):
    """_summary_   output the model weight.

    Args:
        ranker (_type_): _description_
        OutputDict (_type_): _description_
    """
    Weight=ranker.OutputWeights()
    if Weight is not None:
        for ind,weight_i in enumerate(Weight):
            OutputDict[str(ind)+"th-weight"].append(weight_i)
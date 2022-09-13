import numpy as np
import utils.dataset as dataset
import utils.simulation as sim
import utils.rankings as rnk
import utils.evaluation as evl
from collections import defaultdict
from progressbar import progressbar
import argparse
from str2bool import str2bool
import json
import os
import random
import sys
import BatchExpLaunch.results_org as results_org
import time
from algorithms.DBGD.tddbgd import TD_DBGD
from algorithms.PDGD.pdgd import PDGD,PDGDv1
from algorithms.DBGD.tdmgd  import TD_MGD
from algorithms.L2Ranker.NNTopK import NNTopK
from algorithms.L2Ranker.UCBRank import UCBRank
from algorithms.L2Ranker.NNRandomK import NNRandomK
from algorithms.L2Ranker.EBRank import EBRank
from algorithms.L2Ranker.NNEpsilon import NNEpsilon
from algorithms.L2Ranker.EBRankOnlyPrior import EBRankOnlyPrior
from algorithms.L2Ranker.EBRankOnlyBehaviour import EBRankOnlyBehaviour
from algorithms.L2Ranker.BM25Ranker  import BM25
from algorithms.L2Ranker.EBRankSum import EBRankSum
from algorithms.L2Ranker.EBRankV1 import EBRankV1
from algorithms.L2Ranker.NNPriorFeature import NNPriorFeature
from algorithms.L2Ranker.TS import TS
import torch
torch.set_default_dtype(torch.float64)
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str,
                        default="localOutput/",
                        help="Path to result logs")
    parser.add_argument("--dataset_name", type=str,
                        # default="MSLR-WEB30k_beh_rm",
                        default="MQ2007",
                        help="Name of dataset to sample from.")
    parser.add_argument("--dataset_info_path", type=str,
                        default="LTRlocal_dataset_info.txt",
                        help="Path to dataset info file.")
    parser.add_argument("--fold_id", type=int,
                        help="Fold number to select, modulo operator is applied to stay in range.",
                        default=1)
    parser.add_argument("--query_least_size", type=int,
                        default=5,
                        help="query_least_size, filter out queries with number of docs less than this number.")
    parser.add_argument("--queryMaximumLength", type=int,
                    default=np.inf,
                    help="the Maximum number of docs")
    parser.add_argument("--rankListLength", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
    parser.add_argument("--fairness_strategy", type=str,
                        choices=['FairCo', 'FairCo_multip.',"onlyFairness", 'GradFair',"Randomk","FairK",\
                            "ExploreK","Topk","FairCo_maxnorm","QPFair","QPFair-Horiz.","ILP","LP"],
                        default="Randomk",
                        help="fairness_strategy, available choice is ['FairCo', 'FairCo_multip.', 'QPFair','GradFair','Randomk','Topk']")
    parser.add_argument("--fairness_tradeoff_param", type=float,
                            default=0.5,
                            help="fairness_tradeoff_param")
    parser.add_argument("--relvance_strategy", type=str,
                        choices=['TrueAverage',"NNmodel","EstimatedAverage"],
                        default="EstimatedAverage",
                        help="relvance_strategy, available choice is ['TrueAverage', 'NNmodel.', 'EstimatedAverage']")
    parser.add_argument("--exploration_strategy", type=str,
                        choices=['MarginalUncertainty',None],
                        default='MarginalUncertainty',
                        help="exploration_strategy, available choice is ['MarginalUncertainty', None]")
    parser.add_argument("--exploreParam", type=float,
                            default=0,
                            help="exploration_tradeoff_param")
    parser.add_argument("--NewItemEnterProb", type=float,
                        default=1.0,
                        help="the probability of a new item unmasked")
    parser.add_argument("--random_seed", type=int,
                    default=1,
                    help="random seed for reproduction")
    parser.add_argument("--positionBiasSeverity", type=int,
                    help="Severity of positional bias",
                    default=1)
    parser.add_argument("--n_iteration", type=int,
                    default=10**5,
                    help="how many iteractions to simulate")
    parser.add_argument("--n_futureSession", type=int,
                    default=100,
                    help="how many future session we want consider in advance, only works if we use QPFair strategy.")
    parser.add_argument("--progressbar",  type=str2bool, nargs='?',
                    const=True, default=True,
                    help="use progressbar or not.")
    parser.add_argument("--ExpandFeature",  type=str2bool, nargs='?',
                    const=False, default=False,
                    help="use progressbar or not.")
    parser.add_argument("--Ranker", type=str,
                        choices=["BM25",'TD_DBGD', 'PDGD',"TD_MGD", 'NNTopK',"NNRandomK","UCBRank",\
                            "EBRank","NNEpsilon","EBRankOnlyPrior","EBRankOnlyBehaviour","EBRankSum","EBRankV1","TS","NNPriorFeature"],
                            default="NNPriorFeature",
                            help="set the ranker")
    # parser.add_argument("--sparseTraining",  type=str2bool, nargs='?',
    #                 const=True, default=True,
    #                 help="use sparseTraining or not.")
    args = parser.parse_args()
    # args = parser.parse_args(args=[]) # for debug
    # load the data and filter out queries with number of documents less than query_least_size.
    argsDict=vars(args)
    data = dataset.get_data(args.dataset_name,
                  args.dataset_info_path,
                  args.fold_id,
                  args.query_least_size,
                  args.queryMaximumLength,
                  relvance_strategy=args.relvance_strategy,\
                  voidFeature=False,
                  rndSeed=args.random_seed,
                  ExpandFeature=args.ExpandFeature,
                  )
    # initialize Ranker
    RankerClass=eval(args.Ranker)
    r_args=RankerClass.default_parameters()
    r_args["n_results"]=args.rankListLength
    r_args["n_features"]=data.train.feature_matrix.shape[-1]
    r_args["exploreParam"]=args.exploreParam
    r_args["data"]=data
    r_args["LogPath"]=args.log_dir
    ranker=RankerClass(**r_args)
    # Initilization before simulation.
    Logging=results_org.getLogging()
    positionBias=sim.getpositionBias(args.rankListLength,args.positionBiasSeverity) 
    NDCGcutoffs=[i for i in [1,2,3,4,5,10,20,30,50] if i<=args.rankListLength]
    assert args.rankListLength<=args.query_least_size, print("For simplicity, the ranked list length should be greater than doc length")
    queryRNG=np.random.default_rng(args.random_seed) 
    StartqueryRNG=np.random.default_rng(args.random_seed) 
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    clickRNG=np.random.default_rng(args.random_seed) 
    OutputDict=defaultdict(list)
    OnlineResultDict=defaultdict(list)
    NumStartIteration=int(data.train.num_queries()+data.test.num_queries()+data.validation.num_queries())*20
    evalIterationsOrig=np.linspace(0, args.n_iteration-1, num=21,endpoint=True).astype(np.int32)
    evalIterations=evalIterationsOrig[1:]
    UpdateIterations=np.linspace(0, args.n_iteration-1, num=21,endpoint=True).astype(np.int32)[1:]
    iterationsGenerator=progressbar(range(args.n_iteration)) if args.progressbar else range(args.n_iteration)
    StartiterationsGenerator=progressbar(range(NumStartIteration)) if args.progressbar else range(NumStartIteration)
    start_time = time.time()
    n_testCounter=0
    InitialRanker=BM25(**r_args)
    ### initial collection of behaviour using BM25
    for iteration in StartiterationsGenerator:
        # sample data split and a query
        qid,dataSplit=sim.sample_queryFromdata(data,StartqueryRNG)
        ranking=InitialRanker.get_query_dataSplit_LoggingRanking(qid,dataSplit)
        # update exposure statistics according to ranking
        clicks=rnk.simClickForDatasplit(qid,dataSplit,ranking,positionBias,StartqueryRNG)
        dataSplit.updateStatistics(qid,clicks,ranking,positionBias)
    ### simulation of ranking services.
    for iteration in iterationsGenerator:
        # sample data split and a query
        qid,dataSplit=sim.sample_queryFromdata(data,queryRNG)
        dataSplit.UnMaskItems(args.NewItemEnterProb,queryRNG,qid,TimeStamp=iteration) ## Let an item come in given some probability.
        ranking=ranker.get_query_dataSplit_LoggingRanking(qid,dataSplit)
        # update exposure statistics according to ranking
        clicks=rnk.simClickForDatasplit(qid,dataSplit,ranking,positionBias,clickRNG)
        dataSplit.updateStatistics(qid,clicks,ranking,positionBias)
        evl.UpdateOnlineMultipleCutoffsDatasplit(clicks,ranking,qid,dataSplit,positionBias,NDCGcutoffs,OnlineResultDict)
        if iteration in evalIterations:
            Logging.info("current iteration"+str(iteration))
            OutputDict["iterations"].append(iteration)
            OutputDict["time"].append(time.time()-start_time)
            evl.outputCumulative(OnlineResultDict,OutputDict)
            evl.LogStatisticsDatasplit(data,OutputDict)
            evl.evaluate_offlineNDCGDatasplit(ranker,data,positionBias,NDCGcutoffs,OutputDict,Cold=True)
            evl.evaluate_offlineNDCGDatasplit(ranker,data,positionBias,NDCGcutoffs,OutputDict,Cold=False)
            evl.ModelWeightEval(ranker,OutputDict)
        # get a ranking according to fairness strategy
        UpdateLoggingOrNot=iteration in UpdateIterations
        ranker.update_to_interaction(clicks)
        if UpdateLoggingOrNot:
            ranker.updateLoggingWithLearning(data)
        # calculate metrics ndcg and unfairness.
    #write the results.
    least_label=1.0
    result_time_stamp_ctr=evl.time_stamp_clicksDataSplit(data.test,least_label,evalIterationsOrig)
    OutputDict["result_time_stamp_ctr_least_label"+str(least_label)]=result_time_stamp_ctr 
    os.makedirs(args.log_dir,exist_ok=True)
    logPath=args.log_dir+"/result.jjson"
    print('Writing results to %s' % OutputDict)
    with open(logPath, 'w') as f:
        json.dump(OutputDict, f)
        
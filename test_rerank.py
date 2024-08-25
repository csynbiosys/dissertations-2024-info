#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging

import torch
from layers import Model
from torch_geometric.sampler import BaseSampler
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DynamicBatchSampler
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import expit
from torch_geometric.nn.pool import knn_graph
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import json


import random

df_inv_avg_prob = pd.read_csv('investor_avg_likelihood.csv')
df_sig_categories = pd.read_csv('significant_categories.csv')
df_inv_avg_general_prob = pd.read_csv('investor_all_category_avg_likelihood.csv')
df_startup_category_group = pd.read_csv('startup_category_group.csv')

def chunks(lst, n):
    _ = []    

    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])
        
    return _

def split_dataset(x0, x1, edge_index, prop = 0.01):
    d = HeteroData()
    d['inv'].x = torch.FloatTensor(x0)
    d['org'].x = torch.FloatTensor(x1)
    
    d['inv', 'ii', 'org'].edge_index = torch.LongTensor(edge_index)
    
    return d

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None.npz")
    
    parser.add_argument("--lr", default = "2.053595531306313e-06") #0.00012156299936369672
    parser.add_argument("--weight_decay", default = "0.08337675808893397") #.0021852322399311197
    parser.add_argument("--k", default = "22") #15
    
    parser.add_argument("--model_path", default = "gat2_embandfeat_best_model.pth")#model_best_check.pth

    parser.add_argument("--odir", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args
def precision_at_k(actual, predicted, b):
    
    relevant_items = len(set(actual) & set(predicted))
    
    return relevant_items / b

def recall_at_k(actual, predicted, b):
    
    relevant_items = len(set(actual) & set(predicted))
    
    total_positive = len(actual)
    
    return relevant_items / total_positive

def hit_rate(actual, predicted, b):
    
    common = set(actual) & set(predicted)
    if len(common) > 0:
        hit = 1
    elif len(common) == 0:
        hit = 0
    
    return hit


def main():
    
    
    ############# PARAMETERS ##############
    batch_size = 52
    threshold_list = [0.4,0.5,0.6,0.7,0.8,0.9]
    b_list = [5,10,15,20]
    near_zero = 0.00001
    rerank_number_list = [10,20,25,30,35,40,45,50]#,60,70,80,90,100]
    #near_zero_list = [0.00000001, 0.000001, 0.0001, 0.001, 0.005, 0.01, 0.05]

    #######################################
    results = []
    #for x in rerank_number_list:
        #rerank_number = x
    for x in threshold_list:
        threshold  = x
        for x in rerank_number_list:
            rerank_number =  x
        #for x in b_list:
            #rerank_number = 20
            b = 10
            args = parse_args()
            device = torch.device('cpu')
            PATH = args.model_path
            #torch.save(model.state_dict(), PATH)
            x0 = np.load(args.ifile)['x0']
            x1 = np.load(args.ifile)['x1']
            

                
            with np.load('ts_np.npz', allow_pickle=True) as loaded:
                test_orgs = loaded['test_orgs']
                neg_pos = loaded['neg_pos']
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(np.array([np.mean(neg_pos)])).to(device))

            edge_index = np.load('data.npz')['edge_index'][::-1,:].copy()
            
            data = split_dataset(x0, x1, edge_index)
            data = T.ToUndirected()(data)
            
            model  = Model().to(device)
            model.load_state_dict(torch.load(PATH))
            model.eval()
            
            test_losses = []
            test_accs = []
            
            x0 = data['inv'].x
            x1 = data['org'].x

            
            test_orgs = list(test_orgs.item())
            
            edge_index = data['inv', 'ii', 'org'].edge_index
            
            ii = list(range(x0.shape[0]))
            #random.shuffle(ii)
            
            inv_ii = chunks(list(ii), batch_size)
            org_ii = chunks(list(test_orgs), batch_size)
            
            d_sub = dict()
            d_sub['inv'] = torch.LongTensor(np.array(range(x0.shape[0])))
            d_sub['org'] = torch.LongTensor(np.array(test_orgs[:4 * batch_size]))
            
            target = data.subgraph(d_sub)
            
            y = np.zeros((4 * batch_size, x0.shape[0]))
            y[target['inv', 'ii', 'org'].edge_index[1], target['inv', 'ii', 'org'].edge_index[0]] = 1.
            
            logging.info('have {} batches to test against...'.format(len(inv_ii)))
            logging.info('have {} new org batches...'.format(len(org_ii)))
            
            Y_pred = []
            for ix, ii in enumerate(inv_ii):
                if ix % 200 == 0:
                    logging.info('on batch {}...'.format(ix))
                
                # get the graph to test against
                x_inv = x0[ii]
                
                # print(ii)
                
                _, edge_index_, _, _ = k_hop_subgraph(torch.LongTensor(np.array(ii)), 1, edge_index, directed = False, flow = 'target_to_source')
                org_ii_ = list(set(list(edge_index_[1].numpy())))
                        
                org_ii_ = list(set(org_ii_).difference(test_orgs))
            
                d_sub = dict()
                d_sub['inv'] = torch.LongTensor(np.array(ii))
                d_sub['org'] = torch.LongTensor(np.array(org_ii_))
                
                batch = data.subgraph(d_sub)
                
                _ = []
                for ii_ in org_ii[:4]:
                    # print(ii_)
                    x_new = x1[ii_]
                    #x_new = x1[ii_] New oeganizaiton features
            
            
                    # make unified edges and nodes
                    e = batch['inv', 'ii', 'org'].edge_index.clone()
            
                    n_known_edges = e.shape[1]
            
                    e[1] += batch['inv'].x.shape[0]
                    
                    x = torch.cat([batch['inv'].x, batch['org'].x, x_new], 0).to(device)
            
                    knn_indices = knn_graph(torch.cat([batch['org'].x, x_new], 0), int(args.k))
                    knn_indices += batch['inv'].x.shape[0]
            
                    e = torch.cat([e, knn_indices], 1).to(device)
                    
                    edge_attr = torch.zeros((e.shape[1], 2))
                    edge_attr[-knn_indices.shape[1]:,1] = 1.
                    edge_attr[:n_known_edges,0] = 1.
            
                    edge_attr = edge_attr.to(device)
            
                    with torch.no_grad():
                        y_pred = model(x, e, edge_attr, batch['inv'].x.shape[0], x_new.shape[0])
                    
                    _.extend(y_pred.detach().cpu().numpy().T)
                    
                Y_pred.append(np.array(_))
                        
            Y_pred = np.concatenate(Y_pred, 1)

            np.savez('test_preds.npz', y_pred = Y_pred, y = y, ii = np.array(test_orgs[:4 * batch_size], dtype = np.int32))
            
            f = f1_score(y.flatten(), np.round(expit(Y_pred).flatten()))
            
            
            
            hit_rates = []
            precisions = []
            recalls = []
            

            
            Y_pred = torch.FloatTensor(expit(Y_pred)).to(device)
            y = torch.FloatTensor(y).to(device)
            
            loss = criterion(Y_pred, y).item()
            
            
            
            data_pred = np.load('test_preds.npz')
            y_pred = data_pred['y_pred']
            #print(y_pred.shape)
            y = data_pred['y']
            #print(y.shape)
            original_indices = data_pred['ii']
            original_indices = original_indices.tolist()
            #print(original_indices)
            
            with open("investor_name_dict.json", "r") as file:
                investor_name_dict = json.load(file)
                
            with open("startup_name_dict.json", "r") as file:
                startup_name_dict = json.load(file)
            
            top_b_investors_for_startups = {}
            top_rerank_investors_for_startups = {}
            rerank_top_b_investors_for_startups = {}
            
            
            
            for startup_index in range(y_pred.shape[0]):
                
                startup_name = startup_name_dict[str(original_indices[startup_index])]
                
                investor_indices_with_high_predictions = np.where(expit(y_pred[startup_index]) > threshold)[0]
            
                
                sorted_investor_indices = investor_indices_with_high_predictions[np.argsort(-y_pred[startup_index][investor_indices_with_high_predictions])]
                prediction_score = np.argsort(-y_pred[startup_index][investor_indices_with_high_predictions])
                prediction_score = prediction_score.tolist()
                prediction_score = sorted(prediction_score, reverse=True)
                top_rerank_investors = sorted_investor_indices[:rerank_number]
                top_rerank_score = prediction_score[:rerank_number]
                
                top_b_investors = sorted_investor_indices[:b]
            
                
                original_startup_index = original_indices[startup_index]
                top_b_investors_for_startups[original_startup_index] = [investor_name_dict[str(index)] for index in top_b_investors]
                top_rerank_investors_for_startups[original_startup_index] = [investor_name_dict[str(index)] for index in top_rerank_investors]
                #top_b_recommended_investor_names = top_b_investors_for_startups[original_indices[startup_index]]
                top_rerank_recommended_investor_names = top_rerank_investors_for_startups[original_indices[startup_index]]

                if df_startup_category_group[df_startup_category_group['name'] == startup_name]['category_groups_list'].empty:
                    startup_category = ''
                else:
                    startup_category = df_startup_category_group[df_startup_category_group['name'] == startup_name]['category_groups_list'].values[0]
                if isinstance(startup_category, str):
                    categories = [category.strip() for category in startup_category.split(',')]
                else:
                    categories = ['']
                updated_investor_score_dict = {}
                investor_score_dict = dict(zip(top_rerank_recommended_investor_names, top_rerank_score))
                if categories == ['']:
                    continue
                else:
                    for category in categories:
                        assert category in df_inv_avg_general_prob.columns 
                        for name, score in investor_score_dict.items():
                            likelihood = df_inv_avg_general_prob[df_inv_avg_general_prob['investor_name'] == name][category]
                            if likelihood.empty:
                                updated_investor_score_dict[name] = score * near_zero
                            else:
                                updated_investor_score_dict[name] = score * likelihood.values[0]
                    sorted_dict = dict(sorted(updated_investor_score_dict.items(), key=lambda item: item[1], reverse=True))
                    #rerank_investors_100 = list(sorted_dict.keys())[:100]
                    rerank_investors_b = list(sorted_dict.keys())[:b]
                    
                    # SIGNIFICANT CATEGORY BOOST RERANK - ONLY TRIGGER IF STARTUP IS POPULAR CATEGORY GROUP
                    if any(category in df_sig_categories.iloc[-1].values for category in categories):
                        assert len(top_rerank_recommended_investor_names) == len(top_rerank_score)
                        for name, score in investor_score_dict.items():
                            avg_likelihood = df_inv_avg_prob[df_inv_avg_prob['investor_name'] == name]['avg_likelihood']
                            if avg_likelihood.empty:
                                updated_investor_score_dict[name] = score * near_zero
                            else:
                                updated_investor_score_dict[name] = score * avg_likelihood.values[0]
                        sorted_dict = dict(sorted(updated_investor_score_dict.items(), key=lambda item: item[1], reverse=True))
                        rerank_investors = list(sorted_dict.keys())[:rerank_number]
                        rerank_investors_b = list(sorted_dict.keys())[:b]
                        
                        
                    rerank_top_b_investors_for_startups[original_indices[startup_index]] = rerank_investors_b
                        
            for startup_index in range(y.shape[0]):
                
                actual_investor_indices = np.where(y[startup_index] == 1)[0]
                actual_investor_names = [investor_name_dict[str(index)] for index in actual_investor_indices]
                original_startup_index = original_indices[startup_index]
                
                recommended_investor_names = top_b_investors_for_startups[original_indices[startup_index]]
                if original_startup_index not in rerank_top_b_investors_for_startups:
                    rerank_top_b_investors_for_startups[original_startup_index] = recommended_investor_names
                reranked_investor_names = rerank_top_b_investors_for_startups[original_startup_index]
                
                common_investors = set(actual_investor_names).intersection(recommended_investor_names)
                reranked_common_investors = set(actual_investor_names).intersection(reranked_investor_names)
                
                startup_name = startup_name_dict[str(original_indices[startup_index])]
                #print(f"Startup: {startup_name}")
                #print(f"Actual Investors: {actual_investor_names}")
                #print(f"Recommended Investors: {recommended_investor_names}")
                #print(f"RERANKED Recommended Investors: {reranked_investor_names}")
                #if common_investors:
                #    print(f"Common Investors: {common_investors}")
                #if reranked_common_investors:
                #    print(f"RERANKED Common Investors: {reranked_common_investors}")
                #else:
                #    print(f"No common investors between actual and recommended.")
                #print("-----")
            
                
                hit_rates.append(hit_rate(actual_investor_names, reranked_investor_names, b))
                precisions.append(precision_at_k(actual_investor_names, reranked_investor_names, b))
                recalls.append(recall_at_k(actual_investor_names, reranked_investor_names, b))
            #print(f"WITHOUT RERANK:{top_b_investors_for_startups}")
            #print(f"RERANK:{rerank_top_b_investors_for_startups}")
            avg_hit_rate = np.mean(hit_rates)
            avg_precision_at_k = np.mean(precisions)
            avg_recall_at_k = np.mean(recalls)
            logging.info('got eval f1 of {}...'.format(f))
            logging.info(f'Average Hit Rate at {b}: {avg_hit_rate}')
            logging.info(f'Average Precision at {b}: {avg_precision_at_k}')
            logging.info(f'Average Recall at {b}: {avg_recall_at_k}')
            logging.info('got eval loss of {}...'.format(loss))
                    
            #results.append({'Rerank Number' : rerank_number, 'At Number':b, 'Average Hit Rate': avg_hit_rate, 'Average Precision': avg_precision_at_k, 'Average Recall': avg_recall_at_k, 'Eval Loss': loss, 'F1 Score':f})
            #results.append({'Near Zero Value' : near_zero, 'At Number':b, 'Average Hit Rate': avg_hit_rate, 'Average Precision': avg_precision_at_k, 'Average Recall': avg_recall_at_k, 'Eval Loss': loss, 'F1 Score':f})
            results.append({'Threshold' : threshold, 'Rerank Number':rerank_number, 'Average Hit Rate': avg_hit_rate, 'Average Precision': avg_precision_at_k, 'Average Recall': avg_recall_at_k, 'Eval Loss': loss, 'F1 Score':f})
            
            
    df_results = pd.DataFrame(results)
    df_results.to_csv('rerank_threshold_fix_rerank_results.csv', index=False)
    
if __name__ == '__main__':
    main()




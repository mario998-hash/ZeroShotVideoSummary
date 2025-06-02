#hyperParamSearch_TGVS

import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
import sys 
sys.path.append('/root/vidSum')
import numpy as np
from src.model.heuristic_prediction import heuristic_predicator_v5
from semantic_evaluation import generate_summary, load_videos_tag
from semantic_evaluation import calculate_semantic_matching_all

def fetch_gt_summaris(gt_dir, splits, mapping):
    gt_summaries = {}
    for split in splits:
        test_keys = split['test_keys']
        for key in test_keys:
            gt_file_name = mapping[key]['gt_file'] 
            video_id = mapping[key]['video_id'] 
            gt_file = f'{gt_dir}/{video_id}/{gt_file_name}' 
            with open(gt_file, 'r') as f:
                gt_summary = [int(line.strip()) for line in f if line.strip()]
                gt_summary = [x-1 for x in gt_summary]
            gt_summaries[key] = gt_summary

    return gt_summaries

def run(args):

    work_dir = args.work_dir
    splits_file = args.splits_file
    mapping_file = args.mapping_file
    Tags_file = args.Tags_file
    gt_dir = args.gt_dir
    meta_data_dir = args.meta_data_dir
    metrics_output_file = args.metrics_output_file
    shot_metric = args.shot_metric

    video_shots_tag = load_videos_tag(mat_path=Tags_file)
    
    NORM = args.norm
    metric_dir = f'{work_dir}/Eval_{shot_metric}_{NORM}'
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    ## init params search range
    p1_range = range(0, 101, 10)

    print(meta_data_dir)
    if not os.path.exists(meta_data_dir):
        raise ValueError("Meta data file not found !")

    splits = None
    with open(splits_file, 'r') as json_file:
        splits = json.load(json_file)

    mapping = None
    with open(mapping_file, 'r') as json_file:
        mapping = json.load(json_file)

    gt_summaries = fetch_gt_summaris(gt_dir, splits, mapping)
    
    # merge predicted meta data
    meta_data_files = os.listdir(meta_data_dir)
    meta_data_files = [file.split('.')[0] for file in meta_data_files]

    data = {}
    for vidQry in meta_data_files:
        with open(f'{meta_data_dir}/{vidQry}.json','r') as json_file:
            data[vidQry] = json.load(json_file)
        



    # init results dict for ploting 
    header = ['Video ID', 'Sigma', 'Norm', 'Precision', 'Recall', 'Test_F1']
    results = [header]
    
    search_results = {}
    for p1_ in p1_range:
            p1 = p1_ / 100
            for n in [NORM]:
                key = (p1, n)
                splits_f1_scores = []

                for split in splits:

                    test_keys = split['test_keys']
            
                    #test_videos = [mapping[test_k] for test_k in test_keys]

                    test_data = {key: data[key] for key in test_keys}

                    output_file_train = work_dir + f'/{args.config}_output_file_eval.json'
                    heuristic_predicator_v5(test_data, output_file_train, p1, 1-p1, alpha=0, norm=n)
                    
                    gt_summaries_lens = {gt_k : len(gt_summaries[gt_k]) for gt_k in test_data.keys()}
                    machine_summaries = generate_summary(output_file_train, gt_summaries_lens, shot_metric)
                    
                    gt_summaries_tmp = {gt_k : gt_summaries[gt_k] for gt_k in machine_summaries.keys()}
                    _,_, split_f1_score = calculate_semantic_matching_all(machine_summaries, gt_summaries_tmp, video_shots_tag, split['test_video_id']-1)
                    
                    splits_f1_scores.append(split_f1_score)
                    print(split_f1_score)

                search_results[key] = np.mean(splits_f1_scores)
                print(f'{key} : {search_results[key]}')

    best_p1, best_norm = max(search_results, key=search_results.get)

    for split in splits:
        
        print(f"Split : {split['test_video_id']}")
        test_keys = split['test_keys']
        
        #test_videos = [mapping[test_k] for test_k in test_keys]

        test_data = {key: data[key] for key in test_keys}

        output_file_test = work_dir + f'/{args.config}_output_file_test.json'
        heuristic_predicator_v5(test_data, output_file_test, best_p1, 1-best_p1, alpha=0,norm=best_norm)
        
        gt_summaries_lens = {gt_k : len(gt_summaries[gt_k]) for gt_k in test_data.keys()}
        machine_summaries = generate_summary(output_file_test, gt_summaries_lens, shot_metric)
        gt_summaries_tmp = {gt_k : gt_summaries[gt_k] for gt_k in machine_summaries.keys()}
        precision, recall, f1 = calculate_semantic_matching_all(machine_summaries, gt_summaries_tmp, video_shots_tag, split['test_video_id']-1)

        #Header:['Video ID', 'Sigma', 'Norm', 'Precision', 'Recall', 'Test_F1'] 
        state = [split['test_video_id'], best_p1, best_norm, precision, recall, f1]
        results.append(state)
        

    
    # clean up
    if os.path.exists(work_dir + f'/{args.config}_output_file_eval.json'):
        os.remove(work_dir + f'/{args.config}_output_file_eval.json')

    if os.path.exists(work_dir + f'/{args.config}_output_file_test.json'):
        os.remove(work_dir + f'/{args.config}_output_file_test.json')


    # plot metrics
    test_f1_scores = np.array([x[-1] for x in results[1:]])
    mean_f1_scores = np.mean(test_f1_scores)
    std_f1_scores = np.std(test_f1_scores)
    avg = [0]*len(header)
    avg[-1] = mean_f1_scores
    results.append(avg)
    # save results
    metrics_output_file1 = metric_dir + '/' + metrics_output_file
    with open(metrics_output_file1, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(results)

    results_file = metric_dir + f'/{NORM}.json'
    with open(results_file,'w') as json_file:
        data = {k[0] : search_results[k] for k in search_results.keys()}
        json.dump(data, json_file, indent=4)

    print('\n')
    print(f'Sigma = {best_p1} | Norm = {best_norm}')
    print(f'Average F1-score : {mean_f1_scores} ± {std_f1_scores}\n')

    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hyper Param Search GFVS')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--splits_file", type=str, help="videos directory")
    parser.add_argument("--mapping_file", type=str)
    parser.add_argument("--Tags_file", type=str, help='Tags.mat file path')
    parser.add_argument("--gt_dir", type=str, help='Ground truth file path')
    parser.add_argument("--shot_metric",type=str, choices=['mean','max'])
    parser.add_argument("--norm", type=int, choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    parser.add_argument("--meta_data_dir", type=str, help="videos directory")
    parser.add_argument("--metrics_output_file", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    run(args)


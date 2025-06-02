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
from src.model.heuristic_prediction import heuristic_predicator
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
    shot_metric = args.shot_metric
    Norm= args.norm

    video_shots_tag = load_videos_tag(mat_path=Tags_file)
    metric_dir = f'{work_dir}/Eval_{shot_metric}_{Norm}'
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)



    splits = None
    with open(splits_file, 'r') as json_file:
        splits = json.load(json_file)

    mapping = None
    with open(mapping_file, 'r') as json_file:
        mapping = json.load(json_file)

    gt_summaries = fetch_gt_summaris(gt_dir, splits, mapping)
    
    # merge predicted meta data
    meta_data_files = os.listdir(meta_data_dir + '/PredMetaData_1')
    meta_data_files = [file.split('.')[0] for file in meta_data_files]

    data = {}
    for vidQry in meta_data_files:
        data[vidQry] = {i : {} for i in range(1,5,1)}

    for vidQry in meta_data_files:
        for i in range(1,5,1):
            with open(f'{meta_data_dir}/PredMetaData_{i}/{vidQry}.json','r') as json_file:
                data[vidQry][i] = json.load(json_file)
        
    i = 0
    splits_f1_score = []
    for split in splits:
        print(f"Split : {split['test_video_id']}")
        test_keys = split['test_keys']
        
        #test_videos = [mapping[test_k] for test_k in test_keys]

        test_data = {key: data[key] for key in test_keys}

        output_file_test = work_dir + f'/_output_file_test.json'
        #heuristic_predicator_v5(test_data, output_file_test, Sigma, 1-Sigma, alpha=0,norm=NORM)
        heuristic_predicator(test_data, output_file_test, norm=Norm)

        gt_summaries_lens = {gt_k : len(gt_summaries[gt_k]) for gt_k in test_data.keys()}
        machine_summaries = generate_summary(output_file_test, gt_summaries_lens, shot_metric)
        gt_summaries_tmp = {gt_k : gt_summaries[gt_k] for gt_k in machine_summaries.keys()}
        precision, recall, f1 = calculate_semantic_matching_all(machine_summaries, gt_summaries_tmp, video_shots_tag, split['test_video_id']-1)
        splits_f1_score.append(f1)

    if os.path.exists(work_dir + f'/_output_file_test.json'):
        os.remove(work_dir + f'/_output_file_test.json')

    print(np.mean(splits_f1_score))
    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='QFVS evaluation')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--splits_file", type=str, help="videos directory")
    parser.add_argument("--mapping_file", type=str)
    parser.add_argument("--Tags_file", type=str, help='Tags.mat file path')
    parser.add_argument("--gt_dir", type=str, help='Ground truth file path')
    parser.add_argument("--shot_metric",type=str, choices=['mean','max'], default='mean')
    parser.add_argument("--meta_data_dir", type=str, help="videos directory")
    parser.add_argument("--norm",type=str, choices=['None', 'MinMax', 'Exp','MinMax+Exp'])
    args = parser.parse_args()
    run(args)


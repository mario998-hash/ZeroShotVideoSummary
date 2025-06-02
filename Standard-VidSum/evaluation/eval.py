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
from src.evaluation.evaluator import evaluate_summaries
from src.model.heuristic_prediction import heuristic_predicator



def run(args):

    work_dir = args.work_dir
    video_name = args.video_name
    gt_file = args.gt_file
    splits_file = args.splits_file
    mapping_file = args.mapping_file
    metric = args.metric
    meta_data_dir = args.meta_data_dir


    splites = None
    with open(splits_file, 'r') as json_file:
        splites = json.load(json_file)

    with open(mapping_file, 'r') as json_file:
        mapping = json.load(json_file)

    # merge predicted meta data
    meta_data_files = os.listdir(meta_data_dir + '/PredMetaData_1')
    meta_data_files = [file.split('.')[0] for file in meta_data_files]

    data = {}
    for video_name in meta_data_files:
        data[video_name] = {i:{} for i in range(1,5,1)}
    for video_name in meta_data_files:
        for i in range(1,5,1):
            with open(f'{meta_data_dir}/PredMetaData_{i}/{video_name}.json','r') as json_file:
                data[video_name][i] = json.load(json_file)


    Norm = args.norm
    # Eval
    results_all = []
    for i, split in enumerate(splites):
        print(f'Split {i+1}:')
        test_keys = split['test_keys']
        
        test_videos = [mapping[test_k] for test_k in test_keys]

        test_data = {key: data[key] for key in test_videos}

        output_file_test = work_dir + f'/eval.json'

        heuristic_predicator(test_data, output_file_test, norm=Norm)
                    
        split_res,_,_ = evaluate_summaries(output_file_test, gt_file, test_keys, mapping, metric, test=True)
        results_all.append(split_res)
        
    
    # clean up
    if os.path.exists(work_dir + f'/eval.json'):
        os.remove(work_dir + f'/eval.json')

    print(f'Results for {metric} dataset :')
    print(f'F1-score : {np.mean(results_all)}')

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper Param Search')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--video_name", type=str, default="")
    parser.add_argument("--gt_file", type=str, help='Ground truth file path')
    parser.add_argument("--splits_file",type=str)
    parser.add_argument("--mapping_file", type=str, help="videos directory")
    parser.add_argument("--meta_data_dir", type=str, help="videos directory")
    parser.add_argument("--metric",type=str, choices=['summe', 'tvsum'])
    parser.add_argument("--norm",type=str, choices=['None', 'MinMax', 'Exp','MinMax+Exp'])
    args = parser.parse_args()
    run(args)


"""
python /root/vidSum/src/evaluation/eval.py \
--work_dir /root/sumMe_L1In \
--gt_file /root/data/eccv16_dataset_summe_google_pool5.h5 \
--splits_file /root/vidSum/splits/summe_splits_5.json \
--mapping_file /root/vidSum/data_scripts/sumMe_mapping.json \
--meta_data_dir /root/sumMe_L1Out \
--metric summe \
--norm 1
"""
"""
python /root/vidSum/src/evaluation/eval.py \
--work_dir /root/tvSum_L1In \
--gt_file /root/data/eccv16_dataset_tvsum_google_pool5.h5 \
--splits_file /root/vidSum/splits/tvsum_splits_5.json \
--mapping_file /root/vidSum/data_scripts/tvSum_mapping.json \
--meta_data_dir /root/tvSum_L1Out \
--metric tvsum \
--norm 20
"""


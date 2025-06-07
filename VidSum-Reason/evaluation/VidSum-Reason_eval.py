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
from src.evaluation.knapsack_implementation import knapSack
from src.evaluation.evaluation_metrics import evaluate_summary_fscore


def segment_video(N, fragment_size):
    frames_per_segment = max(1, int((fragment_size / 100) * N))  # at least 1 frame
    segments = []
    for start in range(0, N, frames_per_segment):
        end = min(start + frames_per_segment, N)
        segments.append((start, end-1))  # segment is [start, end)

    return np.array(segments)

def generate_summary(scores, fragment_size, summary_size):
    scores = np.array(scores)
    n_frames = scores.shape[0]
    shot_bound = segment_video(n_frames, fragment_size)
    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    shot_scores = []
    shot_lengths = []
    portaion = summary_size/100
    for shot in shot_bound:
        shot_lengths.append(shot[1]-shot[0]+1)
        shot_scores.append((scores[shot[0]:shot[1]+1].mean()).item())
    
        # Select the best shots using the knapsack implementation
        final_max_length = int((shot[1]+1)*portaion)# knapsack budget
        selected = knapSack(final_max_length, shot_lengths, shot_scores, len(shot_lengths))
        
        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(shot[1]+1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1]+1] = 1

    return summary

category_f1_score = {}
def evaluate_summary_VidSumReason(frames_score_file, gt_data, fragment_size, summary_size):
    split_f1_score = []
    with open(frames_score_file, 'r') as json_file:
        machine_scores = json.load(json_file)

    
    for test_key in machine_scores.keys():
        cat = gt_data[test_key]['query_category']
        if cat not in category_f1_score.keys():
            category_f1_score[cat] = []
        gt_scores = gt_data[test_key]['gtscore']
        gt_summary = generate_summary(gt_scores, fragment_size, summary_size)
        predicted_score = machine_scores[test_key]
        machine_summary = generate_summary(predicted_score, fragment_size, summary_size)
        f1_score = evaluate_summary_fscore(machine_summary, np.array([gt_summary]), eval_method='avg')


        category_f1_score[cat].append(f1_score)
        split_f1_score.append(f1_score)

    mean_f1 = np.mean(split_f1_score)

    return mean_f1


def run(args):

    work_dir = args.work_dir
    splits_file = args.splits_file
    gt_dir = args.gt_dir
    meta_data_dir = args.meta_data_dir
    Norm = args.norm
    FragmentSize = args.fragment_size
    SummaryPortaion = args.summary_portion
    

    config = f'@FS{FragmentSize}@SP{SummaryPortaion}'
    if not os.path.exists(meta_data_dir):
        raise ValueError("Meta data file not found !")

    splites = None
    with open(splits_file, 'r') as json_file:
        splites = json.load(json_file)

    # merge predicted meta data
    meta_data_files = os.listdir(meta_data_dir+'/PredMetaData_1')
    meta_data_files = [file.split('.')[0] for file in meta_data_files]

    data = {}
    for video_name in meta_data_files:
        data[video_name] = {i:{} for i in range(1,5,1)}
    gt_data = {}
    for vidQry in meta_data_files:
        for i in range(1,5,1):
            with open(f'{meta_data_dir}/PredMetaData_{i}/{vidQry}.json','r') as json_file:
                data[vidQry][i] = json.load(json_file)

        with open(f'{gt_dir}/{vidQry}.json','r') as json_file:
            gt_data[vidQry] = json.load(json_file)


    splits_f1_scores = []
    for i, split in enumerate(splites):

        test_keys = split['test_keys']
            
        #test_videos = [mapping[test_k] for test_k in test_keys]

        test_data = {key: data[key] for key in test_keys}

        output_file_test = work_dir + f'/{config}_output_file_eval.json'
        heuristic_predicator(test_data, output_file_test, norm=Norm)
         
        split_f1_score = evaluate_summary_VidSumReason(output_file_test, gt_data, FragmentSize, SummaryPortaion)

        splits_f1_scores.append(split_f1_score)

    # clean up
    if os.path.exists(work_dir + f'/{config}_output_file_eval.json'):
        os.remove(work_dir + f'/{config}_output_file_eval.json')

    print('Avg. VidSum-Reason accuracy:')
    print(np.mean(splits_f1_scores))
    print('Avg. category accuracy:')
    for cat_key in category_f1_score.keys():
        print(cat_key)
        print(np.mean(category_f1_score[cat_key]))
    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VidSum-Reason evaluation')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--splits_file", type=str, help="videos directory")
    parser.add_argument("--gt_dir", type=str, help='Ground truth file path')
    parser.add_argument("--meta_data_dir", type=str, help="videos directory")
    parser.add_argument("--norm",type=str, choices=['None', 'MinMax', 'Exp','MinMax+Exp'])
    parser.add_argument("--fragment_size", type=int,default=3)
    parser.add_argument("--summary_portion", type=int,default=36)    
    args = parser.parse_args()

    run(args)



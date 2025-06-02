#PoR_evaluation

import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os
import csv
import sys 
sys.path.append('/root/vidSum')
import numpy as np
from src.evaluation.knapsack_implementation import knapSack
from src.evaluation.evaluation_metrics import evaluate_summary_fscore

def generate_random_scores(vq, gt_dir):
    gt_file = f'{gt_dir}/{vq}.json'
    with open(gt_file, 'r') as file:
        data = json.load(file)
    n_frames = data['n_frames']
    return np.random.rand(n_frames)


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


def evaluate_summary(random_scores, gt_scores, fragment_size, summary_size):
    gt_summary = generate_summary(gt_scores, fragment_size, summary_size)
    machine_summary = generate_summary(random_scores, fragment_size, summary_size)
    f1_score = evaluate_summary_fscore(machine_summary, np.array([gt_summary]), eval_method='avg')

    return f1_score


def plot_results(plot_4results, metric_dir):
    results = []
    for (fs, ss), f1_score in plot_4results.items():
        results.append([fs, ss, f1_score])

    df = pd.DataFrame(results, columns=['Fragment Size(%)', 'Summary Portaion(%)', 'F1-score (mean)'])
    heatmap_data = df.pivot(index='Fragment Size(%)', columns='Summary Portaion(%)', values='F1-score (mean)')
    plt.figure(figsize=(15,15))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2f")
    plt.title(f'PoR Evaluation')
    plt.savefig(f"{metric_dir}/PoR_evaluation.png")

import matplotlib.cm as cm
import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_hist(mapping, gt_dir, metric_dir):
    data = {}
    for key in mapping.keys():
        file = f'{gt_dir}/{key}.json'
        with open(file, 'r') as json_file:
            video_data = json.load(json_file)
            query_category = video_data['query_category']
        data[query_category] = data.get(query_category, 0) + 1

    items = list(data.keys())
    frequencies = list(data.values())

    colors = cm.get_cmap('tab10', len(items))
    color_mapping = {item: colors(i) for i, item in enumerate(items)}

    plt.figure(figsize=(10, 6))

    for idx, (item, freq) in enumerate(zip(items, frequencies)):
        plt.barh(idx, freq, color=color_mapping[item])

    # Remove y-axis tick labels
    plt.yticks(range(len(items)), [''] * len(items))

    # Add legend at the upper right inside the plot
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[item]) for item in items]
    plt.legend(handles, items, title='Query categories', loc='upper right', bbox_to_anchor=(0.95, 0.95),fontsize=12, title_fontsize=13)
    plt.xticks(fontsize=16)  
    plt.xlabel('Frequency', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # grid on x-axis only
    plt.tight_layout()

    os.makedirs(metric_dir, exist_ok=True)
    plt.savefig(f"{metric_dir}/CategoryFreq.png")
    plt.savefig(f"{metric_dir}/CategoryFreq.pdf")
    plt.close()


def run(args):

    work_dir = args.work_dir
    gt_dir = args.gt_dir
    mapping_file = args.mapping_file

    metric_dir = f'{work_dir}/PoR'
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    with open(mapping_file, 'r') as json_file:
        mapping = json.load(json_file)


    data = {}
    for key in mapping.keys():
        gt_file = f'{gt_dir}/{key}.json'
        with open(gt_file, 'r') as file:
            data[key] = json.load(file)


    fragment_size = args.fragment_size
    summary_portion = args.summary_portion
    
    F1_scores = []
    for seed in range(100):
        np.random.seed(seed)
        for vq in mapping.keys():
            n_frames = data[vq]['n_frames']
            random_scores = np.random.rand(n_frames)
            f1_score = evaluate_summary(random_scores, data[vq]['gtscore'], fragment_size, summary_portion) 
            F1_scores.append(f1_score)

    PoR_mean = np.mean(F1_scores)
    print(f'Fragment size : {fragment_size} || Summry portain : {summary_portion}')
    print(f'Mean F1-score : {PoR_mean}')
                
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper Param Search')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--gt_dir", type=str, help='Ground truth file path')
    parser.add_argument("--mapping_file", type=str)
    parser.add_argument("--fragment_size", type=int)
    parser.add_argument("--summary_portion", type=int)

    args = parser.parse_args()
    run(args)


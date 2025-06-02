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
from src.model.heuristic_prediction import heuristic_predicator_v5

def plot_results(test_f1_scores, plot_4results, work_dir, metric, config):

    # Assuming test_f1_scores is a list or numpy array
    splits_num = range(1, len(test_f1_scores) + 1)
    mean_f1 = np.mean(test_f1_scores)
    median_f1 = np.median(test_f1_scores)

    plt.plot(splits_num, test_f1_scores, label="Test F1-scores", marker='o')
    plt.axhline(y=mean_f1, color='r', linestyle='--', label=f"Mean F1 ({mean_f1:.2f})")
    plt.axhline(y=median_f1, color='g', linestyle=':', label=f"Median F1 ({median_f1:.2f})")

    plt.xlabel('Split Number')
    plt.ylabel('Test F1-score')
    plt.title(f'{metric} results')
    plt.legend()
    plt.grid(True)  # Optional: Adds grid lines for better visualization
    plt.savefig(f'{work_dir}/{config}/{metric}_TestF1.png')
    

    # heat maps 
    means, median, maxs, mins = [], [], [], []
    for (st, dt), split_res in plot_4results.items():
        means.append([st, dt, np.mean(split_res).item()])
        maxs.append([st, dt, np.max(split_res).item()])
        mins.append([st, dt, np.min(split_res).item()])
        median.append([st, dt, np.median(split_res).item()])

    ress = [means, median, maxs, mins]
    titles = ['mean', 'median', 'max', 'min']
    for res, title in zip(ress, titles):
        df = pd.DataFrame(res, columns=['p1', 'p2', 'Value'])
        heatmap_data = df.pivot(index='p1', columns='p2', values='Value')
        plt.figure(figsize=(30,22))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2f")
        plt.title(f'{metric} Test F1 {title}')
        plt.savefig(f"{work_dir}/{config}/{metric}_TestF1_{title}.png")


def run(args):

    work_dir = args.work_dir
    splits_file = args.splits_file
    gt_file = args.gt_file
    mapping_file = args.mapping_file
    metric = 'summe' if 'summe' in str(splits_file).lower() else 'tvsum'
    meta_data_dir = args.meta_data_dir
    metrics_output_file = args.metrics_output_file
    norm = args.NORM

    metric_dir = f'{work_dir}/ALPHA_SIGMA_v5/EVAL_{norm}'
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    ## init params search range
    p1_vals = list(map(float, args.sigma_range))
    min_p1 = p1_vals[0]
    max_p1 = p1_vals[1] 
    p1_step = p1_vals[2]
    p1_range = np.arange(min_p1, max_p1, p1_step)
    p2_range = p1_range
    
    p3_range = np.arange(0.0, 1.001, 0.1)

    print(meta_data_dir)
    if not os.path.exists(meta_data_dir):
        raise ValueError("Meta data file not found !")

    splites = None
    with open(splits_file, 'r') as json_file:
        splites = json.load(json_file)

    with open(mapping_file, 'r') as json_file:
        mapping = json.load(json_file)

    # merge predicted meta data
    meta_data_files = os.listdir(meta_data_dir)
    meta_data_files = [file.split('.')[0] for file in meta_data_files]

    data = {}
    for video_name in meta_data_files:
        with open(f'{meta_data_dir}/{video_name}.json','r') as json_file:
            data[video_name] = json.load(json_file)


    # init results dict for ploting 
    header = ['Split Index', 'Sigma', 'Alpha', 'Test_F1']
    results = [header]
    
    search_results = {}
    for p1 in p1_range:
            for p3 in p3_range:
                key = (p1,p3)
                splits_f1_scores = []
                for i, split in enumerate(splites):

                    test_keys = split['test_keys']
            
                    test_videos = [mapping[test_k] for test_k in test_keys]

                    test_data = {key: data[key] for key in test_videos}

                    output_file_train = work_dir + f'/{args.config}_{norm}_output_file_eval.json'
                    heuristic_predicator_v5(test_data, output_file_train, p1, 1-p1, p3, norm=norm)
                    

                    split_f1_score, _, _ = evaluate_summaries(output_file_train, gt_file, test_keys, mapping, metric)

                    splits_f1_scores.append(split_f1_score)

                search_results[key] = np.mean(splits_f1_scores)
                print(f'{key} : {search_results[key]}')

    best_p1, best_p3 = max(search_results, key=search_results.get)
    
    # Eval
    for i, split in enumerate(splites):
        print(f'Split {i+1}:')
        test_keys = split['test_keys']
        
        test_videos = [mapping[test_k] for test_k in test_keys]

        test_data = {key: data[key] for key in test_videos}

        output_file_test = work_dir + f'/{args.config}_{norm}_output_file_test.json'
        heuristic_predicator_v5(test_data, output_file_test, best_p1, 1-best_p1, best_p3, norm=norm)
        

        test_f1_score, _, _ = evaluate_summaries(output_file_test, gt_file, test_keys, mapping, metric, test=True)

        
        #Header:['Split Index', 'p1', 'p3', 'Test_F1']
        state = [i+1, best_p1, best_p3, test_f1_score]
        results.append(state)
        

    
    # clean up
    if os.path.exists(work_dir + f'/{args.config}_{norm}_output_file_eval.json'):
        os.remove(work_dir + f'/{args.config}_{norm}_output_file_eval.json')

    if os.path.exists(work_dir + f'/{args.config}_{norm}_output_file_test.json'):
        os.remove(work_dir + f'/{args.config}_{norm}_output_file_test.json')


    # plot metrics
    test_f1_scores = np.array([state[-1] for state in results[1:]])
    mean_f1_scores = np.mean(test_f1_scores)
    std_f1_scores = np.std(test_f1_scores)
    avg = [0]*len(header)
    avg[-1] = mean_f1_scores
    results.append(avg)
    # save results
    with open(metrics_output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(results)


    print('\n')
    print(f'Sigma = {best_p1} | sAlpha = {best_p3}')
    print(f'Results for {metric} dataset :')
    print(f'Average F1-score : {mean_f1_scores}  {std_f1_scores}\n')

    return 


if __name__ == "__main__":
    def str2Range(arg):
        return arg.split('-')
    parser = argparse.ArgumentParser(description='Hyper Param Search')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--splits_file", type=str, help="videos directory")
    parser.add_argument("--gt_file", type=str, help='Ground truth file path')
    parser.add_argument("--mapping_file", type=str)
    parser.add_argument("--meta_data_dir", type=str, help="videos directory")
    parser.add_argument("--metrics_output_file", type=str)
    parser.add_argument("--sigma_range", type=str2Range)
    parser.add_argument("--config", type=str)
    parser.add_argument("--NORM", type=int, choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], default=0)
    args = parser.parse_args()
    run(args)
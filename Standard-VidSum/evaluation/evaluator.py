import numpy as np
import json
import h5py
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary_fscore, evaluate_summary_fscore_all

def evaluate_summaries(frames_score_file, gt_file, keys, mapping, metric, test=False):
    assert metric == 'summe' or metric == 'tvsum', 'Error with args in evaluator/evaluate_summaries/metric'

    if not gt_file.endswith('.h5'):
        raise ValueError("Ground truth file must be a 'h5' type !")
        
    # load gt data
    hdf = h5py.File(gt_file, 'r')
    all_user_summary = []
    shot_bound =[]
    n_frames = []
    
    for video_index in list(keys):# {video_1, video_2, ... etc}
        summary = np.array(hdf.get(video_index + '/user_summary') )
        sb = np.array(hdf.get(video_index + '/change_points') )
        video_frames_num = np.array(hdf.get(video_index + '/n_frames'))


        all_user_summary.append(summary)
        shot_bound.append(sb)
        n_frames.append(int(video_frames_num))
    
    hdf.close()

    # generate summarise
    with open(frames_score_file, 'r') as json_file:
        data = json.load(json_file)

    gt_keys = [mapping[video_index] for video_index in list(keys)] 
    videos_frame_scores = [data[key] for key in gt_keys]

    all_machine_summarise = generate_summary(shot_bound, videos_frame_scores, n_frames)

    # evaluate metrics
    eval_method = 'max' if metric == 'summe' else 'avg'# tvsum

    F1_score = []
    for machine_summary, summary in zip(all_machine_summarise, all_user_summary):
        video_f1_score = evaluate_summary_fscore(machine_summary, summary, eval_method)
        F1_score.append(video_f1_score)

    return np.mean(F1_score).item(), all_machine_summarise, F1_score


def evaluate_summaries_all(frames_score_file, gt_file, keys, mapping, metric, test=False):
    assert metric == 'summe' or metric == 'tvsum', 'Error with args in evaluator/evaluate_summaries/metric'

    if not gt_file.endswith('.h5'):
        raise ValueError("Ground truth file must be a 'h5' type !")
        
    # load gt data
    hdf = h5py.File(gt_file, 'r')
    all_user_summary = []
    shot_bound =[]
    n_frames = []
    
    for video_index in list(keys):# {video_1, video_2, ... etc}
        summary = np.array(hdf.get(video_index + '/user_summary') )
        sb = np.array(hdf.get(video_index + '/change_points') )
        video_frames_num = np.array(hdf.get(video_index + '/n_frames'))


        all_user_summary.append(summary)
        shot_bound.append(sb)
        n_frames.append(int(video_frames_num))
    
    hdf.close()

    # generate summarise
    with open(frames_score_file, 'r') as json_file:
        data = json.load(json_file)

    gt_keys = [mapping[video_index] for video_index in list(keys)] 
    videos_frame_scores = [data[key] for key in gt_keys]

    all_machine_summarise = generate_summary(shot_bound, videos_frame_scores, n_frames)

    # evaluate metrics
    eval_method = 'max' if metric == 'summe' else 'avg'# tvsum

    results = []
    for machine_summary, summary in zip(all_machine_summarise, all_user_summary):
        split_res = evaluate_summary_fscore_all(machine_summary, summary, eval_method)
        results.append(split_res)

    return np.mean(results,axis=0)
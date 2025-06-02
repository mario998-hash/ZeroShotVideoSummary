import json
import torch
import torch.nn.functional as F
import numpy as np
import ot
device = "cuda" if torch.cuda.is_available() else "cpu"
########## V5 ##########
# With SIN for smooth transition and the option to descrease the difference between consactive scene scores by a factor of ALPHA
def weighted_scores_v2(data, tuned_frames_score, w_consistency, w_dissimilarity, norm):
    frames_consistency = data['consistency']
    frames_dissimilarity = data['dissimilarity']
    n_frames = min(len(tuned_frames_score), len(frames_consistency))
    query_relevance = [1.0] *  n_frames
    if 'frame_query_correlation' in data.keys():
        query_relevance = data['frame_query_correlation']

    tuned_frames_score = tuned_frames_score[:n_frames]
    frames_consistency = frames_consistency[:n_frames]
    frames_dissimilarity = frames_dissimilarity[:n_frames]
    query_relevance = query_relevance[:n_frames]

    frames_scores = []
    for i, scene_score in enumerate(tuned_frames_score):
        consistency_score = frames_consistency[i]
        dissimilarity_score = frames_dissimilarity[i]
        weighted_consistency = w_consistency * consistency_score
        # WAS : weighted_dissimilarity = w_dissimilarity * (1-dissimilarity_score)
        weighted_dissimilarity = w_dissimilarity * (dissimilarity_score)
                    
        # The contribution score is the weighted sum of the consistency and dissimilarity scores
        contribution_score = weighted_consistency + weighted_dissimilarity
        frame_score = scene_score * contribution_score
        frames_scores.append(frame_score)

    frames_start = data['scene_frames']
    normed_scores = [0]*len(frames_scores)
    for (start, end) in zip(frames_start[:-1], frames_start[1:]):
        curr_scores = frames_scores[start:end]
        curr_scores = Norm_scores(curr_scores, norm)
        #curr_query_corr = query_relevance[start:end]
        #curr_query_corr = Norm_scores(curr_query_corr, norm)

        #curr_scores = np.array(curr_scores)
        #curr_query_corr = np.array(curr_query_corr)

        #curr_normed_score = curr_scores * curr_query_corr
        #curr_scores = curr_normed_score.tolist()
        normed_scores[start:end] = curr_scores

    return normed_scores


def Norm_scores(array_, norm_config):
    eps = 1e-8
    arr = np.array(array_)

    if norm_config == 0:
        # No scaling, return as is
        return arr.tolist()

    if norm_config == 1:
        # Min-Max scaling [0, 1]
        array = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + eps)
        return array.tolist()

    if norm_config == 2:
        # Z-score scaling + Sigmoid transformation [0, 1]
        z_scores = (arr - np.mean(arr)) / (np.std(arr) + eps)
        array = 1 / (1 + np.exp(-z_scores))  # Sigmoid transformation
        return array.tolist()

    if norm_config == 3:
        # Mean Normalization and shift to [0, 1]
        array = (arr - np.mean(arr)) / (np.max(arr) - np.min(arr) + eps) + 0.5
        return array.tolist()

    if norm_config == 4:
        # Robust Scaling using Median and IQR
        median = np.median(arr)
        q1 = np.percentile(arr, 25)  # 25th percentile
        q3 = np.percentile(arr, 75)  # 75th percentile
        iqr = q3 - q1
        if iqr == 0:
            return arr.tolist()
        # Apply Robust Scaling
        array = (arr - median) / (iqr + eps)
        return array.tolist()

    if norm_config == 5:
        array = np.exp(arr / 30)
        return array.tolist()

    if norm_config == 6:
        array = np.exp(arr / 40)
        return array.tolist()

    if norm_config == 7:
        array = np.exp(arr / (np.mean(arr) + eps))
        return array.tolist()

    if norm_config == 8:
        q1 = np.percentile(arr, 25)  # 25th percentile
        if q1 == 0:
            return arr.tolist()

        array = np.exp(arr / (q1 + eps))
        return array.tolist()

    if norm_config == 9:
        median = np.median(arr)
        if median == 0:
            return arr.tolist()

        array = np.exp(arr / (median + eps))
        return array.tolist()

    if norm_config == 10:
        q2 = np.percentile(arr, 20)  # 20th percentile
        if q2 == 0:
            return arr.tolist()

        array = np.exp(arr / (q2 + eps))
        return array.tolist()

    if norm_config == 11:
        min_ = np.min(arr)
        if min_ == 0:
            return arr.tolist()

        array = np.exp(arr / (min_ + eps))
        return array.tolist()

    if norm_config == 12:
        array = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + eps)
        array = np.exp(array)
        return array.tolist()

    if norm_config == 13:
        q1 = np.percentile(arr, 25)  # 25th percentile
        if q1 == 0:
            return arr.tolist()
        exp_array = np.exp(arr / (q1 + eps))
        array = [x / sum(exp_array) for x in exp_array]
        return array

    if norm_config == 14:
        array = (arr - np.min(arr)) / (np.mean(arr) - np.min(arr) + eps)
        return array.tolist()

    if norm_config == 15:
        exp_x = np.exp(arr - np.max(arr))
        array = exp_x / np.sum(exp_x)
        return array.tolist()


    if norm_config == 16:
        # L2 Norm (Euclidean norm)
        l2_norm = np.linalg.norm(arr)
        if l2_norm == 0:
            return arr.tolist()
        array = arr / (l2_norm + eps)
        return array.tolist()

    if norm_config == 17:
        # L1 Norm (Manhattan norm)
        l1_norm = np.sum(np.abs(arr))
        if l1_norm == 0:
            return arr.tolist()
        array = arr / (l1_norm + eps)
        return array.tolist()

    if norm_config == 18:
        # Log(x + 1) Transformation
        array = np.log(arr + 1 + eps)  # Log(x + 1) to avoid log(0)
        return array.tolist()

    if norm_config == 19:
        # Rank Transformation
        ranks = np.argsort(np.argsort(arr)) + 1  # Assign ranks to each value
        array = ranks / np.max(ranks)  # Normalize ranks between 0 and 1
        return array.tolist()
    
    if norm_config == 20:
        array = np.exp(arr/100)
        return array.tolist()
    
    if norm_config == 21:
        array = np.exp(arr)
        return array.tolist()


def smooth_transition_v2(video_data, alpha, norm):
    frames_start = video_data['scene_frames']
    scenes_scores = video_data['scene_scores']
    scenes_scores = Norm_scores(scenes_scores, norm)


    expanded_scores = []
    for i in range(1,len(frames_start)):
        curr = [scenes_scores[i-1]]*(frames_start[i] - frames_start[i-1])
        expanded_scores.extend(curr)

    
    expanded_scores = np.array(expanded_scores)  
    num_scenes = len(scenes_scores)

    for i in range(1, num_scenes):
        start1 = frames_start[i-1]
        start2 = frames_start[i]
        start3 = frames_start[i+1]
    
        scene1_half1 = (start2 - start1)//2
        scene1_half2 = (start3 - start2)//2

        transition_length = scene1_half1 + scene1_half2 + 1
        transition_x = np.linspace(0, np.pi, transition_length)

        x1, x2 = scenes_scores[i-1], scenes_scores[i]
        diff = abs(x1 - x2)/2
        # Sine-based interpolation for smooth transition
        if x1 < x2:
            x1 += alpha*diff
            x2 -= alpha*diff
         
            transition = x1 + (x2 - x1) * (1 - np.cos(transition_x)) / 2  # Smooth increase
        else:
            x2 += alpha*diff
            x1 -= alpha*diff
            
            transition = x1 - (x1 - x2) * (1 - np.cos(transition_x)) / 2  # Smooth decrease

        expanded_scores[start1  + scene1_half1 : start1 + scene1_half1 + transition_length] = transition  # Apply transition


    return expanded_scores.tolist()


def heuristic_predicator_v5(data, frames_score_file, w_consistency, w_dissimilarity, alpha, norm=0):
    
    # compute the segments score based on the heuristic
    scores = {}
    for video_name in data.keys():
        tuned_frames_score =  smooth_transition_v2(data[video_name], alpha, norm)
        frames_score = weighted_scores_v2(data[video_name], tuned_frames_score, w_consistency, w_dissimilarity, norm)
        scores[video_name] = frames_score


    with open(frames_score_file, 'w') as json_file:
        json.dump(scores, json_file, indent=4)

    return 


def calc_sigma(data,mid_video=1.8):
    
    total_frames = data[1]['n_frames']
    fps = data[1]['video_fps']
    video_duration = (total_frames/fps)/60

    if video_duration > 5*mid_video:
        return (0.1, 1)
    
    if video_duration > mid_video :# < 10 on 4 good
        return (1.0, 1)
    
    return (0.3, 3)
    

def heuristic_predicator_v6(data, frames_score_file, norm):    
    # compute the segments score based on the heuristic
    scores = {}
    for video_name in data.keys():
        sigma, w = calc_sigma(data[video_name])
        tuned_frames_score =  smooth_transition_v2(data[video_name][w], alpha=0, norm=norm)
        w_consistency = sigma
        w_dissimilarity = 1 -sigma
        frames_score = weighted_scores_v2(data[video_name][w], tuned_frames_score, w_consistency, w_dissimilarity, norm)
        scores[video_name] = frames_score


    with open(frames_score_file, 'w') as json_file:
        json.dump(scores, json_file, indent=4)

    return

###############################################################

def select_sigma_w(data):
    mid_video = 1.8 #minutes

    total_frames = data[1]['n_frames']
    fps = data[1]['video_fps']
    video_duration = (total_frames/fps)/60 # in minutes

    if video_duration > 5*mid_video:
        return (0.1, 1)
    
    if video_duration > mid_video :
        return (1.0, 1)
    
    return (0.3, 3)
    

def norm_func(array_, norm_config):
    eps = 1e-8
    arr = np.array(array_)

    if norm_config == 'None':
        # No scaling
        return arr.tolist()

    if norm_config == 'MinMax':
        # Min-Max scaling [0, 1]
        array = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + eps)
        return array.tolist()

    if norm_config == 'Exp':
        # division by 100 is added to prevent value overflowing
        array = np.exp(arr/100)
        return array.tolist()

    if norm_config == 'MinMax+Exp':
        array = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + eps)
        array = np.exp(array)
        return array.tolist()

    else:
        ValueError('Normalization methods isn\'t supported!')
    
    return None


def temporal_smoothing_func(video_data, norm):
    frames_start = video_data['scene_frames']
    scenes_scores = video_data['scene_scores']
    scenes_scores = norm_func(scenes_scores, norm)


    expanded_scores = []
    for i in range(1,len(frames_start)):
        curr = [scenes_scores[i-1]]*(frames_start[i] - frames_start[i-1])
        expanded_scores.extend(curr)

    
    expanded_scores = np.array(expanded_scores)  
    num_scenes = len(scenes_scores)

    for i in range(1, num_scenes):
        start1 = frames_start[i-1]
        start2 = frames_start[i]
        start3 = frames_start[i+1]
    
        scene1_half1 = (start2 - start1)//2
        scene1_half2 = (start3 - start2)//2

        transition_length = scene1_half1 + scene1_half2 + 1
        transition_x = np.linspace(0, np.pi, transition_length)

        x1, x2 = scenes_scores[i-1], scenes_scores[i]
        #diff = abs(x1 - x2)/2
        # Sine-based interpolation for smooth transition
        if x1 < x2:
            transition = x1 + (x2 - x1) * (1 - np.cos(transition_x)) / 2  # Smooth increase
        else:
            transition = x1 - (x1 - x2) * (1 - np.cos(transition_x)) / 2  # Smooth decrease

        expanded_scores[start1  + scene1_half1 : start1 + scene1_half1 + transition_length] = transition  # Apply transition


    return expanded_scores.tolist()


def predict_frame_scores(data, tuned_frames_score, w_consistency, w_dissimilarity, norm):
    frames_consistency = data['consistency']
    frames_dissimilarity = data['dissimilarity']
    n_frames = min(len(tuned_frames_score), len(frames_consistency))

    tuned_frames_score = tuned_frames_score[:n_frames]
    frames_consistency = frames_consistency[:n_frames]
    frames_dissimilarity = frames_dissimilarity[:n_frames]
    query_relevance = query_relevance[:n_frames]

    # frame weights
    frames_scores = []
    for i, scene_score in enumerate(tuned_frames_score):
        consistency_score = frames_consistency[i]
        dissimilarity_score = frames_dissimilarity[i]

        weighted_consistency = w_consistency * consistency_score
        weighted_dissimilarity = w_dissimilarity * (dissimilarity_score)
                    
        # The contribution score is the weighted sum of the consistency and dissimilarity scores
        contribution_score = weighted_consistency + weighted_dissimilarity
        frame_score = scene_score * contribution_score
        frames_scores.append(frame_score)

    # fusing frame weights and smoothend frames scores 
    frames_start = data['scene_frames']
    normed_scores = [0]*len(frames_scores)
    for (start, end) in zip(frames_start[:-1], frames_start[1:]):
        curr_scores = frames_scores[start:end]
        curr_scores = norm_func(curr_scores, norm)
        normed_scores[start:end] = curr_scores

    return normed_scores


def heuristic_predicator(data, frames_score_file, norm):    
    # compute the segments score based on the heuristic
    scores = {}
    for video_name in data.keys():
        sigma, w = select_sigma_w(data[video_name])
        tuned_frames_score =  temporal_smoothing_func(data[video_name][w], norm=norm)
        w_consistency = sigma
        w_dissimilarity = 1 -sigma
        frames_score = predict_frame_scores(data[video_name][w], tuned_frames_score, w_consistency, w_dissimilarity, norm)
        scores[video_name] = frames_score


    with open(frames_score_file, 'w') as json_file:
        json.dump(scores, json_file, indent=4)

    return

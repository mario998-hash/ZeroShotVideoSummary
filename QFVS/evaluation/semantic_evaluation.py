#semantic_evaluation
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import networkx as nx
from sklearn.metrics import pairwise_distances
import numpy as np
import scipy.io
import json

def process_video_mat(video_mat):
    result = []
    for shot_vec in video_mat:
        shot_vec= shot_vec[0][0]
        result.append(shot_vec)
    result = np.array(result)
    return result


def process_mat(mat):
    videos = mat['Tags'][0]
    result = []
    for video_mat in videos:
        video_mat = video_mat[0]
        video_data = process_video_mat(video_mat)
        result.append(video_data)
    return result


def load_videos_tag(mat_path):
    mat = scipy.io.loadmat(mat_path)
    videos_tag = process_mat(mat)
    return videos_tag


def semantic_iou(a, b):
    intersection = a * b
    intersection_num = sum(intersection)
    union = a + b
    union[union>0] = 1
    union_num = sum(union)
    if union_num != 0:
        return intersection_num / union_num
    else:
        return 0


def build_graph_from_pariwise_weights(weight_matrix):
    B = nx.Graph()
    bottom_nodes = list(map(lambda x: "b-{}".format(x), list(range(weight_matrix.shape[0]))))
    top_nodes = list(map(lambda x: "t-{}".format(x), list(range(weight_matrix.shape[1]))))
    edges = []
    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
            weight = weight_matrix[i][j]
            edges.append(("b-{}".format(i), "t-{}".format(j), weight))
    B.add_weighted_edges_from(edges)
    return B


def calculate_semantic_matching(machine_summary, gt_summary, video_shots_tag, video_id):
    video_shots_tag = video_shots_tag[video_id]
    machine_summary_mat = video_shots_tag[machine_summary]
    gt_summary_mat = video_shots_tag[gt_summary]
    weights = pairwise_distances(machine_summary_mat, gt_summary_mat, metric=semantic_iou)
    B = build_graph_from_pariwise_weights(weights)
    matching_edges = nx.algorithms.matching.max_weight_matching(B)
    sum_weights = 0
    i = 0
    for edge in matching_edges:
        edge_data = B.get_edge_data(edge[0], edge[1])
        sum_weights += edge_data['weight']
        i += 1
    precision = sum_weights / machine_summary_mat.shape[0]
    recall = sum_weights / gt_summary_mat.shape[0]
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

"""
Implemented by the paper's author : Mario Barbara
"""

def calculate_semantic_matching_all(machine_summaries, gt_summaries, video_shots_tag, video_id):
    precisions = []
    recalls = []
    f1s = []
    for vidQry in machine_summaries.keys():
        machine_summary = machine_summaries[vidQry]
        gt_summary = gt_summaries[vidQry]
        precision, recall, f1 = calculate_semantic_matching(machine_summary, gt_summary, video_shots_tag, video_id)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
 

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


def frame_to_shot_cast(frames_score, shot_size, shot_metric, FPS=15):
    frames_score = np.array(frames_score)
    frames_per_shot = int(shot_size * FPS)
    num_frames = len(frames_score)
    
    shot_scores = []
    for i in range(0, num_frames, frames_per_shot):
        end = min(i + frames_per_shot, num_frames)
        shot = frames_score[i:end]
        if shot_metric == "mean":
            shot_score = np.mean(shot)
        elif shot_metric == "max":
            shot_score = np.max(shot)
        else:
            raise ValueError(f"Unknown shot_metric: {shot_metric}")
        shot_scores.append(shot_score)
    
    return shot_scores


def construct_budget_summary(shot_scores, Portion):

    N = Portion

    # Get indices of the top-N largest elements
    top_indices = np.argpartition(shot_scores, -N)[-N:]

    # Sort the top indices by their original index order (not value)
    sorted_by_index = np.sort(top_indices)
    return sorted_by_index


def generate_summary(data_file, gt_summaries_lens, shot_metric, SHOT_DURATION=5):
    data = None
    with open(data_file, 'r') as json_file :
        data = json.load(json_file)

    summaries = {}
    for vidQry in data.keys():
        frame_scores = data[vidQry]
        shot_scores = frame_to_shot_cast(frame_scores, SHOT_DURATION, shot_metric)
        if int(vidQry.split('_')[1]) > 135: # video PO4
            shot_scores = shot_scores[:-1]
        summary_len = gt_summaries_lens[vidQry]
        summary = construct_budget_summary(shot_scores, summary_len)
        summaries[vidQry] = summary



    return summaries


if __name__=='__main__':
    video_shots_tag = load_videos_tag(mat_path="./Tags.mat")
    #Test
    machine_summary = [1, 39, 99, 31, 778]
    gt_summary = [1, 34, 101, 29, 774]
    print(calculate_semantic_matching(machine_summary, gt_summary, video_shots_tag, video_id=0))
    

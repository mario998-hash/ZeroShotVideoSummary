#utils
import argparse
import os
import json
# scores prediction
import numpy as np
import torch
import os
import sys 
sys.path.append('/root/vidSum')
from src.utils import *
from embeddings import Embedder

class myModel:

    def __init__(self, args):

        self.video_dir = args.video_dir
        self.work_dir = args.work_dir
        
        self.video_type = args.video_type
        self.video_fps = None
        self.selected_sst = None
        self.n_frames = None
        self.query = None
        self.prediciton_meta_data_file = None
        self.window_size = args.window_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.prediciton_meta_data_dir = self.work_dir + f'/PredMetaData_FLI_{self.window_size}' 
        self.frame_emb_dir = self.work_dir + '/FrameEmb/'

    #
    def set_video_meta_data(self, video_name, VidQry):
        self.VidQry = VidQry
        self.video_name = video_name
        self.video_path = self.video_dir + '/' + self.video_name + '.' + self.video_type
        self.n_frames = get_video_frames_num(self.video_path)
        if self.VidQry != '' :
            self.prediciton_meta_data_file = self.prediciton_meta_data_dir + f'/{self.VidQry}.json'
        else:
            self.prediciton_meta_data_file = self.prediciton_meta_data_dir + f'/{self.video_name}.json'
        self.frame_emb_file = self.frame_emb_dir + f'/{self.video_name}'
        return
    
    #
    def init_pipline_dirs(self, dirs_names):
        for dir_name in dirs_names:
            dir_path = self.work_dir + '/' + dir_name
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)  # Create the directory
                except Exception as e:
                    raise ValueError(f"Error creating directory '{dir_path}': {e}")

    # segment score 
    def calc_frames_data(self, cluster_algo, start_frames, n_frames):
        frames_consistency = []
        frames_dissimilarity = []
        scene_start_frames = start_frames
        if scene_start_frames[-1] < n_frames:
            scene_start_frames.append(n_frames)

        embeddings = np.load(self.frame_emb_file + '.npy')
        
        segments_consistency = []
        segments_dissimilarity = []
        for i, (start, end) in enumerate(zip(scene_start_frames[:-1], scene_start_frames[1:])):
            
            scene_embeddings = embeddings[start : end]
            labels = cluster_algo.automate_clustering(scene_embeddings)
            segments_labels, segment_indcies   = cluster_algo.segment_labels(labels, end-start, self.window_size * self.video_fps)
            
            for segment_labels, segment_index in zip(segments_labels, segment_indcies):
                segment_embeddings = embeddings[start + segment_index[0]: start + segment_index[1]]
                consistency_score, dissimilarity_score = cluster_algo.segment_contribution(segment_labels, segment_embeddings)
                segments_consistency.append((consistency_score, len(segment_labels)))
                segments_dissimilarity.append((dissimilarity_score, len(segment_labels)))

        for score, n_segment in segments_consistency:
            for _ in range(n_segment):
                frames_consistency.append(score)


        for score, n_segment in segments_dissimilarity:
            for _ in range(n_segment):
                frames_dissimilarity.append(score)
         
        assert len(frames_consistency) == n_frames, 'Error in casting segments to frames in Frame Scoring'
        assert len(frames_dissimilarity) == n_frames, 'Error in casting segments to frames in Frame Scoring'

        return frames_consistency, frames_dissimilarity

    #
    def calc_frames_data_SW(self, cluster_algo, start_frames, n_frames):
        frames_consistency = []
        frames_dissimilarity = []
        scene_start_frames = start_frames
        if scene_start_frames[-1] < n_frames:
            scene_start_frames.append(n_frames)
        
        embeddings = np.load(self.frame_emb_file + '.npy')
        for i, (start, end) in enumerate(zip(scene_start_frames[:-1], scene_start_frames[1:])):
            
            final_diversity, final_consistency = self.scene_cont_data(cluster_algo, embeddings[start:end])
            frames_consistency.extend(final_consistency)
            frames_dissimilarity.extend(final_diversity)

        assert len(frames_consistency) == n_frames, 'Error in casting segments to frames in Frame Scoring'
        assert len(frames_dissimilarity) == n_frames, 'Error in casting segments to frames in Frame Scoring'

        return frames_consistency, frames_dissimilarity

    #
    def scene_cont_data(self, cluster_algo,  scene_embeddings):
        num_frames = len(scene_embeddings)
        
        # Initialize arrays to accumulate diversity and consistency scores
        accumulated_diversity = np.zeros(num_frames)
        accumulated_consistency = np.zeros(num_frames)
        frame_counts = np.zeros(num_frames)

        window_size = int(self.window_size * self.video_fps)
        step_size = int(window_size//2)

       
        labels = cluster_algo.automate_clustering(scene_embeddings)
        # Slide the window across the scene
        for start_idx in range(0, num_frames - window_size + 1, step_size):
            
            end_idx = start_idx + window_size
            
            segment_labels = labels[start_idx:end_idx]
            segment_embeddings = scene_embeddings[start_idx:end_idx]
            # Compute diversity and consistency for this window
            diversity, consistency = cluster_algo.segment_contribution(segment_labels, segment_embeddings)

            # Accumulate scores for each frame in the window
            for frame_idx in range(start_idx, end_idx):
                accumulated_diversity[frame_idx] += diversity
                accumulated_consistency[frame_idx] += consistency
                frame_counts[frame_idx] += 1
        
        # Normalize scores for each frame
        final_diversity = np.divide(accumulated_diversity, frame_counts, 
                                    out=np.zeros_like(accumulated_diversity), where=frame_counts > 0)
        
        final_consistency = np.divide(accumulated_consistency, frame_counts, 
                                    out=np.zeros_like(accumulated_consistency), where=frame_counts > 0)
        
        return final_diversity, final_consistency

    # save video meta data 
    def save_results(self, scene_scores, scene_frames, frames_consistency, frames_dissimilarity, frame_query_correlation):
        
        video_prediction_meta_data = {}
        video_prediction_meta_data['video_name'] = self.video_name
        if self.query is not None :
            video_prediction_meta_data['query'] = self.query
        video_prediction_meta_data['video_path'] = self.video_path
        video_prediction_meta_data['video_fps'] = self.video_fps
        video_prediction_meta_data['scene_scores'] = scene_scores
        video_prediction_meta_data['scene_frames'] = scene_frames
        video_prediction_meta_data['n_frames'] = self.n_frames
        video_prediction_meta_data['consistency'] = frames_consistency
        video_prediction_meta_data['dissimilarity'] = frames_dissimilarity
        video_prediction_meta_data['sst'] = self.selected_sst
        video_prediction_meta_data['frame_query_correlation'] = frame_query_correlation

        # cache video's embeddings 
        print(self.prediciton_meta_data_file)
        with open(self.prediciton_meta_data_file, 'w') as json_file:
            json.dump(video_prediction_meta_data, json_file, indent=4)


    def calc_frame_query_corr(self, frames_embeddings: np.ndarray, query_embedding: torch.Tensor):
        """
        Calculates cosine similarity between each frame embedding and the query embedding.

        Args:
            frames_embeddings (np.ndarray): Array of shape (N, D), where N is number of frames.
            query_embedding (torch.Tensor): Tensor of shape (1, D), already normalized.

        Returns:
            List[float]: Similarity scores between each frame and the query.
        """
        # Convert query embedding to numpy
        query_vec = query_embedding.cpu().numpy().squeeze()  # Shape: (D,)

        # Compute dot product (cosine similarity)
        similarities = frames_embeddings @ query_vec  # Shape: (N,)

        return similarities.tolist()


def run(args):


    if args.video_name == "":
        videos_names = list(os.listdir(args.video_dir))
        videos_names = [name.split('.')[0] for name in videos_names if name.endswith('.' + args.video_type)]
    else :
        videos_names = [args.video_name]

    VidQry = args.VidQry
    my_model = myModel(args)
    meta_dir = f'PredMetaData_FLI_{my_model.window_size}' 

    my_model.init_pipline_dirs([meta_dir])
    
    og_PredMetaData_dir = args.og_PredMetaData_dir
    print(og_PredMetaData_dir)
    device = "cuda"
    embedder = Embedder(device)

    for video_name in videos_names:

        my_model.set_video_meta_data(video_name, VidQry)
        if os.path.exists(my_model.prediciton_meta_data_file):
            continue
        torch.cuda.empty_cache()
        print(f'{video_name} Starting ...')
        og_PredMetaData_file = f'{og_PredMetaData_dir}/{VidQry}.json'
        print(og_PredMetaData_file)
        if not os.path.exists(og_PredMetaData_file):
            continue
        
        with open(og_PredMetaData_file, 'r') as json_file:
            data = json.load(json_file)
        

        my_model.video_path = data['video_path']
        my_model.video_fps = data['video_fps']
        scene_scores = data['scene_scores']
        start_frames = data['scene_frames']
        my_model.n_frames = data['n_frames']
        my_model.selected_sst = data['sst']
        frames_consistency = data['consistency']
        frames_dissimilarity = data['dissimilarity']
        if 'query' in data.keys():
            my_model.query = data['query']
            
        frames_embeddings = np.load(my_model.frame_emb_file + '.npy')
        query_embedding = embedder.get_query_embedding(my_model.query)
        frame_query_correlation = my_model.calc_frame_query_corr(frames_embeddings, query_embedding)

        # save results
        my_model.save_results(scene_scores, start_frames, frames_consistency, frames_dissimilarity, frame_query_correlation)

        print(f'{video_name} DONE!')
        
    return 0

if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()


    parser.add_argument("--video_name", type=str, default="")
    parser.add_argument("--video_dir", type=str)
    parser.add_argument("--video_type", type=str, choices=['webm', 'mp4'], default='webm')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--og_PredMetaData_dir", type=str)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--VidQry", type=str, default='', help='Key for the (video, query) pair - results directory')
    args = parser.parse_args()
    run(args)



"""
python /root/vidSum/src/model/query_tune.py \
--video_name wvDENCN4i3c \
--video_dir /root/data/TGVS/videos \
--video_type mp4 \
--work_dir /root/TGVS \
--og_PredMetaData_dir /root/TGVS/PredMetaData_2 \
--window_size 2 \
--VidQry vidQry_2
"""
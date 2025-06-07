#utils
import argparse
import os
import json
# scene detection
import torch
import os
import sys 
import time
sys.path.append('vidSum')
from src.utils import *
# descriptions
from clusters import ClusterFrame
from embeddings import Embedder
from embeddings_dino import Embedder_dino
import openai
from src.model.model import myModel

def solve(args):

    openai.api_key = args.openai_key
    if video_name == '':
        videos_names = list(os.listdir(args.video_dir))
        videos_names = [name.split('.')[0] for name in videos_names if name.endswith('.' + args.video_type)]
    else:

        videos_names = [args.video_name]
    
    user_query = args.user_query
    VidQry = args.VidQry

    if user_query != '':
        assert VidQry != '', 'If user query is provided you should pass a key for the pair (video, query), e.g. VidQry_\{i\}.'
    
    if user_query == '':
        assert VidQry == '', 'Key for the pair (video, query) doesn\'t isn\'t supported when the query isn\'t provided.'
    
    QUERY_PROVIDED = user_query != ''

    device = "cuda"
    my_model = myModel(args)
    my_model.init_pipline_dirs(['sceneDesc', 'FrameEmb'])

    # init embedder
    embedder = Embedder(device)
    # init Clusterer
    cluster_algo = ClusterFrame() 

    for video_name in videos_names:


        my_model.set_video_meta_data(video_name, VidQry)
        if os.path.exists(my_model.prediciton_meta_data_file):
            if QUERY_PROVIDED :
                print(f'Meta Data for the Video-Query (w\ VidQry : {VidQry}) already exites, SKIPING!')
            else :
                print(f'Meta Data for the Video {video_name} already exites, SKIPING!')
            continue

        print(f'{video_name} Starting ...')
        torch.cuda.empty_cache()
        
        #scene detection
        scene_list, start_frames = my_model.detect_scences(None)
        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)

        #cache frame embeddings 
        frames = fetch_frames(my_model.video_path)
        if not os.path.exists(my_model.frame_emb_file + '.npy') :
            embedder.cache_frame_embeddings(my_model.frame_emb_file, frames)

        start_frames = my_model.merge_scenes(start_frames, my_model.frame_emb_file + '.npy', min_frames=150)
        
        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)

        # generate scenes discriptions
        scene_discription_file_name = my_model.generate_scene_descriptions(frames, start_frames)

        scene_scores = my_model.compute_scenes_score(scene_discription_file_name, user_query)

        for W in range(1,5,1):
            my_model.window_size = W
            meta_dir = f'PredMetaData_{my_model.window_size}' 
            my_model.init_pipline_dirs([meta_dir])
            my_model.prediciton_meta_data_dir = my_model.work_dir + f'/PredMetaData_{my_model.window_size}' 
            frames_consistency, frames_dissimilarity = my_model.calc_frames_data(cluster_algo, start_frames, len(frames))

            # save results
            my_model.save_results(scene_scores, start_frames, frames_consistency, frames_dissimilarity, user_query)


        print(f'{video_name} DONE!')

    return 

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="MAIN")
    # Add arguments
    
    parser.add_argument("--video_name", type=str)
    parser.add_argument("--video_type", type=str, choices=['mp4', 'webm'], help='mp4 or webm')

    parser.add_argument("--user_query",type=str, default='')
    parser.add_argument("--VidQry", type=str, default='', help='Key for the (video, query) pair - results directory')
    parser.add_argument("--video_dir", type=str, help="video/s directory")
    
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--description_model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--min_scene_duration", type=int, default=2)
    parser.add_argument("--segment_duration", type=int, default=1)
    parser.add_argument("--openai_key", type=str, default='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')

    args = parser.parse_args()
    solve(args)
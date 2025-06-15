#utils
import argparse
import os
import json
# scene detection
import torch
import os
import sys
sys.path.append('vidSum')
from src.utils import *
# descriptions
from clusters import ClusterFrame
from embeddings import Embedder
from embeddings_dino import Embedder_dino
import openai
from src.model.model import myModel

def run(args):
    openai.api_key = args.openai_key
    
    video_name = args.video_name 
    mapping_file = args.mapping_file

    with open(mapping_file, 'r') as json_file :
        mapping = json.load(json_file)

    device = "cuda"
    my_model = myModel(args)
    my_model.init_pipline_dirs(['sceneDesc', 'FrameEmb'])

    MN_FRAMES = 150
    #og_data = {'P01': 53, 'P02':47, 'P03':49, 'P04':29}

    # init embedder
    embedder = Embedder(device)
    # init Clusterer
    cluster_algo = ClusterFrame() 
    
    my_model.set_video_meta_data(video_name, None)
    #scene detection
    #sst = og_data[video_name]
    scene_list, start_frames = my_model.detect_scences(None)

    if start_frames[-1] < my_model.n_frames:
        start_frames.append(my_model.n_frames)

    #cache frame embeddings 
    frames = fetch_frames(my_model.video_path)
    if not os.path.exists(my_model.frame_emb_file + '.npy') :
        embedder.cache_frame_embeddings_v2(my_model.frame_emb_file, frames)

    
    start_frames = my_model.merge_scenes(start_frames, my_model.frame_emb_file + '.npy', min_frames=MN_FRAMES)
    if start_frames[-1] < my_model.n_frames:
        start_frames.append(my_model.n_frames)

    # generate scenes discriptions
    scene_discription_file_name = my_model.generate_scene_descriptions(frames, start_frames)

    frames_consistency, frames_dissimilarity = my_model.calc_frames_data(cluster_algo, start_frames, len(frames))

    user_queries = []
    for VidQry in mapping.keys() :
        
        if mapping[VidQry]['video_id'] != video_name:
            continue

        torch.cuda.empty_cache()
        my_model.set_video_meta_data(video_name, VidQry)
        if os.path.exists(my_model.prediciton_meta_data_file):
            print(f'Meta Data for the Video-Query (w\ VidQry : {VidQry}) already exites, SKIPING!')
            continue

        print(f'{VidQry} Starting ...')
        user_queries.append(mapping[VidQry]['query'])

    print('Predicting ...')

    queries_scores = my_model.compute_scenes_score_QFVS(scene_discription_file_name, user_queries)

    qry_idx = 0
    for VidQry in mapping.keys() :
        
        if mapping[VidQry]['video_id'] != video_name:
            continue
        user_query = mapping[VidQry]['query']
        scene_scores = queries_scores[qry_idx]
        my_model.set_video_meta_data(video_name, VidQry)
        my_model.save_results(scene_scores, start_frames, frames_consistency, frames_dissimilarity, user_query)
        qry_idx += 1
        print(f'{VidQry} DONE!')
    
 
    return 0

if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="MAIN")
    # Add arguments
    parser.add_argument("--openai_key", type=str)
    parser.add_argument("--video_name", type=str, choices=['P01','P02','P03','P04'])
    parser.add_argument("--video_dir", type=str, help="video/s directory")
    parser.add_argument("--video_type", type=str, choices=['mp4', 'webm'], default='mp4')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--description_model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--min_scene_duration", type=int, default=2)
    parser.add_argument("--mapping_file",type=str)

    args = parser.parse_args()
    run(args)
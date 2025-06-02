import argparse
import sys
sys.path.append('/root/vidSum')
from src.utils import *
import json
import os

def run(args):

    videos_dir = args.videos_dir
    video_name = args.video_name
    user_query = args.user_query
    query_category = args.query_category
    VidQry = args.VidQry
    results_dir = args.results_dir
    results_file = f'{results_dir}/{VidQry}.json'

    if os.path.exists(results_file):
        print('VidQry key aleardy exits! ')
        exit(0)

    video_path = f'{videos_dir}/{video_name}.mp4'
    total_frames = get_video_frames_num(video_path)
    fps = get_video_FPS(video_path)
    
    frames_per_interval = int(2 * fps)  # 2-second window
    frame_scores = [0] * total_frames

    print(f"Video: {video_name}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Frames per 2s: {frames_per_interval}")
    print(f'Assign imporant scores for each segment, score should be in range [1-5], where 5 is representative for the user Query and Vise Versa.\nUser Query : {user_query} .\n')
    i = 1
    for start_frame in range(0, total_frames, frames_per_interval):
        end_frame = min(start_frame + frames_per_interval, total_frames)
        user_score = float(input(f"{i} : Score for frames {start_frame} to {end_frame - 1}: "))
        i += 1
        for f in range(start_frame, end_frame):
            frame_scores[f] = user_score

    # Normalize scores by 5
    normalized_scores = [s / 5.0 for s in frame_scores]
    results_file = f'{results_dir}/{VidQry}.json'
    
    video_data = {
        'video_name' : video_name,
        'user_query' : user_query,
        'query_category' : query_category,
        'video_fps' : fps,
        'n_frames' : total_frames,
        'gtscore' : normalized_scores
    }
    
    with open(results_file, 'w') as json_file:
        json.dump(video_data, json_file, indent=4)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score video frames in 2-second intervals.")
    parser.add_argument("--videos_dir", type=str, default='/root/data/TGVS/videos')
    parser.add_argument("--video_name", type=str, help="Path to the input video file")
    parser.add_argument("--user_query", type=str, required=True)
    parser.add_argument("--query_category", type=str)
    parser.add_argument("--VidQry", type=str, help='Key used for saving the scores for the pair (video, query)')
    parser.add_argument("--results_dir", type=str, default='/root/data/TGVS/GT')
    args = parser.parse_args()

    run(args)

"""
python /root/vidSum/src/TQVS_utils/user_guided_anno_build.py \
--video_name 77AA5bWCGlE \
--user_query "Focus on scenes where the scores changes" \
--query_category "Reasoning with General Knowledge" \
--VidQry vidQry_9
"""





import h5py
import numpy as np
import os
import cv2
import json
import argparse

def get_video_frames_num(video_path):
    cap = cv2.VideoCapture(video_path)
    n = 0
    while True :
        ret, _ = cap.read()
        if not ret:
            break
        n += 1

    cap.release()
    return n

def run(args):    
    videos_dir = args.videos_dir
    n_videos = args.n_videos
    gt_file = args.gt_file
    output_file = args.output_file
    videos_names = [video_name for video_name in sorted(os.listdir(videos_dir)) if video_name.endswith('.mp4')]
    videos_abs_path = [videos_dir + '/' + video_name for video_name in videos_names]

    data_set = []
    ds_frames = []
    for video_name, video_path in zip(videos_names,videos_abs_path):
        video_name = video_name.split('.')[0]# without '.mp4'
        n_frame = get_video_frames_num(video_path)
        ds_frames.append(n_frame)
        data_set.append((video_name, int(n_frame)))

    ds_frames = np.unique(ds_frames)
    assert len(ds_frames) == n_videos, 'There is dups in frames number in the videos'
    # sort videos by frames number
    data_set = sorted(data_set, key= lambda item : item[1])


    hdf = h5py.File(gt_file, 'r')
    gt_set = []
    gt_frames = []
    for video_name in list(hdf.keys()):# {video_1, video_2, ... etc}
        #video_acctual_name = np.array(hdf[video_name + '/video_name']).astype(str).item()
        video_frames_num = np.array( hdf.get(video_name+'/n_frames'))
        gt_frames.append(int(video_frames_num))
        gt_set.append((video_name, int(video_frames_num)))
    
    hdf.close()
    gt_frames = np.unique(gt_frames)
    assert len(gt_frames) == n_videos, 'There is dups in frames number in the videos'
    # sort videos by frames number
    gt_set = sorted(gt_set, key= lambda item : item[1])


    assert len(gt_set) == len(data_set), 'number of videos not matched'
    mapping = {}
    for i in range(n_videos):
        print(f"DS : name - {data_set[i][0]} | nframes - {data_set[i][1]}\n")
        print(f"GT : name - {gt_set[i][0]} | nframes - {gt_set[i][1]}\n")
        print('##############################################')
        #assert gt_n_frames[i][1] == n_frames[i][1], f'Frames doesnt match {i}'
        # mapping video index to video name 
        mapping[gt_set[i][0]] = data_set[i][0]


    # save the mapping
    with open(output_file,'w') as json_file :
        json.dump(mapping, json_file, indent=4)

    return 0
if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser(description='Name&Index mapping')
    parser.add_argument("--videos_dir", type=str, help="videos directory")
    parser.add_argument("--n_videos", type=int, help="number of videos")
    parser.add_argument("--gt_file", type=str, help="gt file path")
    parser.add_argument("--output_file", type=str, help='file path to save the mapping in')
    args = parser.parse_args()
    run(args)

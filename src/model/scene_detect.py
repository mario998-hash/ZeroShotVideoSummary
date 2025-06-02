#utils
import argparse
import os
import json
# scene detection
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips, ColorClip
# scores prediction
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys 
import time
sys.path.append('/root/vidSum')
from src.utils import *
# descriptions
from description_generator import DescriptionGenerator
from clusters import ClusterFrame
from embeddings import Embedder
from embeddings_dino import Embedder_dino
import openai
from sklearn.metrics.pairwise import cosine_similarity

def generate_scene_prompt_v1(video_description, part_description):
    prompt = ""
    prompt += "You are tasked with evaluating the importance of a specific scene within a larger video, considering its role in the overall narrative and message of the video.\n"
    prompt += "I've provided two descriptions below: one for the entire video and one for the specific scene (part) within that video.\n"
    prompt += "Your goal is to assess how critical this particular segment is to the understanding or development of the video's main themes, messages, or emotional impact.\n"
    prompt += "Please assign an importance score on a scale of 1 to 100 for the segment, based on how crucial it is to the overall video. The scale is defined as follows:\n"

    scale = ""
    scale += "1-20 : minimally important (contributes very little to the overall theme or message)\n"
    scale += "21-40 : somewhat important (offers limited context or details that support the main theme)\n"
    scale += "41-60 : moderately important (provides useful context or details that support the main theme)\n"
    scale += "61-80 : quite important (adds significant context or detail that enhances understanding of the main theme)\n"
    scale += "81-100 : highly important (crucial to understanding or conveying the main message of the video)\n"
    prompt += scale

    prompt += "When evaluating, consider factors such as the thematic relevance of the segment, any unique information it provides, and its emotional or narrative impact within the context of the entire video.\n"
    prompt += "Provide your answer with just a number, Don't add explination about the score please or anything else\n"

    prompt += "Whole Video Description: " + video_description + '\n'
    prompt += "Part Description: " + part_description + '\n'
    
    return prompt

def generate_scene_prompt(video_description, part_description):
    prompt = ""
    prompt += "You are tasked with evaluating the importance of a specific scene within a larger video, considering its role in the overall narrative and message of the video. I've provided two descriptions below: one for the entire video and one for the specific scene (part) within that video.\n"
    prompt += "Your goal is to assess how critical this particular segment is to the understanding or development of the video's main themes, messages, or emotional impact. Assign an importance score on a scale of 1 to 100, based on how crucial it is to the overall video. The scale is defined as follows:\n"
    prompt += "* 1-20: Minimally important (contributes very little to the overall theme or message)\n"
    prompt += "* 21-40: Somewhat important (offers limited context or details that support the main theme)\n"
    prompt += "* 41-60: Moderately important (provides useful context or details that support the main theme)\n"
    prompt += "* 61-80: Quite important (adds significant context or detail that enhances understanding of the main theme)\n"
    prompt += "* 81-100: Highly important (crucial to understanding or conveying the main message of the video)\n"
    prompt += "When evaluating, focus on the core narrative or emotional impact of the video. Only assign high scores (80+) to the segments that **directly drive the main theme or message forward**. Be critical and biased towards giving low scores to segments that do not add significant value to the overall narrative or theme. The distribution of high scores should be low and reserved for only the most crucial moments in the video.\n"
    prompt += "The video should be summarized briefly, so please evaluate whether the scene is critical to include in the summary of the video, based on its contribution to the core message. **Prioritize scenes that are essential for a concise summary and omit secondary or supporting moments unless they provide meaningful context.**\n"
    prompt += "Provide only the score in your answer, without any explanation or reasoning.\n"

    prompt += "Whole Video Description: " + video_description + '\n'
    prompt += "Part Description: " + part_description + '\n'
    return prompt

def generate_scene_prompt_all(video_description, total_frames, masked_scenes):
    """
    Generates a prompt to evaluate the importance of multiple  scenes in a video summary,
    incorporating total frame count and per-scene frame counts with start/end boundaries.

    :param video_description: The full description of the video.
    :param total_frames: Total number of frames in the video.
    :param scenes: A list of tuples (scene_number, scene_description, scene_frames, (start_frame, end_frame)).
    :return: A formatted prompt string.
    """

    prompt = (
        "You are tasked with evaluating the importance of specific scenes in a video summary. "
        "Your goal is to assess how much each scene"
        "affects the completeness of the summary.\n\n"

        f"The **total number of frames** in the original video is **{total_frames}**.\n"
        "Each  scene's frame count and its exact **start and end positions** are provided. "
        "Keep in mind that the final summary should only be **15% of the original video size**, "
        "so only the most essential scenes should receive high scores.\n\n"

        "**Evaluation Criteria:**\n"
        "- **Whole Video Description**: A summary of the full video, covering all scenes.\n"
        "- **Scene Descriptions**: Summaries a spacific scene from the video.\n\n"

        "**Scoring Guidelines (1-100) – Be Critical:**\n"
        "- **1-20**: Minimally important (its removal barely affects understanding).\n"
        "- **21-40**: Somewhat important (some context or details lost).\n"
        "- **41-60**: Moderately important (provides useful context, but not crucial).\n"
        "- **61-80**: Quite important (significant context lost, affects the summary).\n"
        "- **81-100**: **Highly important (crucial to understanding the video, should be included in the 15% summary).**\n\n"

        "Consider the **scene's frame count and its position in the video**. A short but **key** scene might be more important than a long, unimportant one.\n\n"

        "**Provide only the scores in the following format (without explanations):**\n"
        "Scene 1: [score]\n"
        "Scene 2: [score]\n"
        "...\n"
        "Scene i: [score]\n"
        "...\n"

        "**Whole Video Description:**\n"
        f"[{video_description}]\n\n"

        "**Scene Descriptions:**\n"
    )

    for scene_number, masked_description, scene_frames, (start_frame, end_frame) in masked_scenes:
        prompt += (
            f"**Scene {scene_number} (Frames: {scene_frames}, Start-End: {start_frame}-{end_frame}):**\n"
            f"[{masked_description}]\n\n"
        )

    return prompt

def init_description_model(description_model_name):
    pretrained = description_model_name
    model_name = "llava_qwen"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = "auto"
    conv_model_template = "qwen_1_5"
    model = DescriptionGenerator(pretrained, model_name, conv_model_template, device=device, device_map=device_map)
    return model

class myModel:

    def __init__(self, args):

        self.video_dir = args.video_dir
        self.work_dir = args.work_dir
        
        self.video_type = args.video_type
        self.video_fps = None
        self.selected_sst = None
        self.n_frames = None
        self.prediciton_meta_data_file = None
        self.min_scene_duration = args.min_scene_duration
        self.window_size = args.window_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.scene_description_dir = self.work_dir + '/sceneDesc'
        self.prediciton_meta_data_dir = self.work_dir + '/PredMetaData'
        self.frame_emb_dir = self.work_dir + '/FrameEmb/'

        self.description_model = init_description_model(args.description_model_name)
        self.batch_size = args.batch_size


        self.scale = 100
        self.meta_data = {}
        self.token_count = 0
        self.TPM = 200_000
        self.gpt_model = "gpt-4o"
        #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

    #
    def set_video_meta_data(self, video_name):
        self.video_name = video_name
        self.video_path = self.video_dir + '/' + self.video_name + '.' + self.video_type
        self.n_frames = get_video_frames_num(self.video_path)
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

    # 
    def detected_scenes_num(self, sst):
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=sst, min_scene_len=self.min_scene_duration*self.video_fps))
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        return len(scene_list)

    # select threhold 
    def scene_threshold_selection(self, sst_range=range(2,71,2), history_n=8):
        lens = []
        acc_range = []
        for h, sst in enumerate(sst_range):
            curr_scene_n = self.detected_scenes_num(sst)
            # early stoping 
            if h >= history_n and np.all([curr_scene_n == x for x in lens[h-history_n:h]]):
                break
            lens.append(curr_scene_n)
            acc_range.append(sst)
            # Plot the second video in the pair if available
        # 
        lens = np.array(lens)
        lens_diff = lens[1:] - lens[:-1]
        mn = np.argmin(lens_diff)
        selected_sst = (acc_range[mn+1] + acc_range[mn])/2
        print(self.video_name)
        print(selected_sst)
        return int(selected_sst)

    # detect scences
    def detect_scences(self, sst=None):

        self.video_fps = get_video_FPS(self.video_path)
        if sst == None :
            self.selected_sst = self.scene_threshold_selection()
        else :
            self.selected_sst = sst
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.selected_sst, min_scene_len=self.min_scene_duration*self.video_fps))

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # List of scenes with start and end timecodes
        scene_list = scene_manager.get_scene_list()
        if len(scene_list) > 0:
            frames_start = [int(self.video_fps * scene[0].get_seconds()) for scene in scene_list]
            return [[scene[0], scene[1]] for scene in scene_list], frames_start
        
        self.selected_sst = 2
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.selected_sst, min_scene_len=self.min_scene_duration*self.video_fps))

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # List of scenes with start and end timecodes
        scene_list = scene_manager.get_scene_list()
        frames_start = [int(self.video_fps * scene[0].get_seconds()) for scene in scene_list]

        return [[scene[0], scene[1]] for scene in scene_list], frames_start

    #
    def merge_scenes(self, scenes_bounds_frames, frame_emb_file, min_frames):
        """
        Merges small scenes based on cosine similarity with neighboring scenes.

        Args:
            scenes_bounds_frames (list): List of scene start frame numbers.
            frame_emb_file (str): File path to the .npy file containing frame embeddings.
            min_frames (int): Minimum number of frames a scene must have to avoid merging.

        Returns:
            list: New list of scene boundaries after merging.
        """
        framesEmd_file = np.load(frame_emb_file, allow_pickle=True)

        while True:
            # Compute the average embedding for each scene
            scene_embeddings = []
            scene_lengths = []
            valid_scenes = []  # Store the scene start frames and their corresponding end frames

            for i in range(len(scenes_bounds_frames) - 1):
                start, end = scenes_bounds_frames[i], scenes_bounds_frames[i + 1]
                frames = framesEmd_file[start:end]  # Directly slice the numpy array
                if frames.size > 0:
                    mean_embedding = np.mean(frames, axis=0)
                else:
                    mean_embedding = np.zeros_like(framesEmd_file[0])  # Ensure valid shape (using the first frame's embedding)

                scene_embeddings.append(mean_embedding)
                scene_lengths.append(end - start)
                valid_scenes.append((start, end))  # Store the scene's start and end

            merged = False
            new_scenes = []  # List to store the updated scene boundaries
            i = 0

            while i < len(scene_embeddings):
                if scene_lengths[i] >= min_frames:
                    new_scenes.append(valid_scenes[i][1])  # Keep the current scene if it's large enough
                    i += 1
                    continue

                # Compute cosine similarity with previous and next scenes (if they exist)
                prev_sim = cosine_similarity(scene_embeddings[i].reshape(1, -1), scene_embeddings[i - 1].reshape(1, -1))[0][0] if i > 0 else -1
                next_sim = cosine_similarity(scene_embeddings[i].reshape(1, -1), scene_embeddings[i + 1].reshape(1, -1))[0][0] if i < len(scene_embeddings) - 1 else -1

                # Merge with the more similar scene
                if prev_sim > next_sim and i > 0:
                    # Merge with the previous scene
                    scene_embeddings[i - 1] = (scene_embeddings[i - 1] * scene_lengths[i - 1] + scene_embeddings[i] * scene_lengths[i]) / (scene_lengths[i - 1] + scene_lengths[i])
                    scene_lengths[i - 1] += scene_lengths[i]
                    merged = True  # A merge occurred
                    i += 1  # Skip the current scene since it's merged with the previous one
                elif next_sim >= prev_sim and i < len(scene_embeddings) - 1:
                    # Merge with the next scene
                    scene_embeddings[i + 1] = (scene_embeddings[i + 1] * scene_lengths[i + 1] + scene_embeddings[i] * scene_lengths[i]) / (scene_lengths[i + 1] + scene_lengths[i])
                    scene_lengths[i + 1] += scene_lengths[i]
                    new_scenes.append(valid_scenes[i + 1][1])  # Corrected to append end frame of merged scene
                    merged = True  # A merge occurred
                    i += 1  # Skip the current scene since it's merged with the next one
                else:
                    # If no merge, keep the current scene
                    new_scenes.append(valid_scenes[i][1])
                    i += 1

            # If no scenes were merged, exit the loop
            if not merged:
                break

            # Update the valid scenes after merging
            scenes_bounds_frames = [valid_scenes[0][0]] + new_scenes  # Update with the new boundaries


        if scenes_bounds_frames[-1] < self.n_frames:
            scenes_bounds_frames.append(self.n_frames)
        while scenes_bounds_frames[-1] - scenes_bounds_frames[-2] < min_frames:
            scenes_bounds_frames.pop(-2)
        while scenes_bounds_frames[1] < min_frames:
            scenes_bounds_frames.pop(1)


        return scenes_bounds_frames


def run(args):


    if args.video_name == "":
        videos_names = list(os.listdir(args.video_dir))
        videos_names = [name.split('.')[0] for name in videos_names if name.endswith('.' + args.video_type)]
    else :
        videos_names = [args.video_name]

    if args.ds == 'tvsum':
        og_data = {'kLxoNp-UchI': 45, 'AwmHb44_ouw': 17, 'VuWGsYPqAX8': 65, 'NyBmCxDoHJU': 55, '3eYKfiOEJNs': 47, 'fWutDQy1nnY': 21, '0tmA_C6XwfM': 13, 'JgHubY5Vw3Y': 5, 'cjibtmSLxQ4': 37, 'z_6gVvQb2d0': 33, 'b626MiF1ew4': 57, 'J0nA4VgnoCo': 15, 'xwqBXPGE9pQ': 5, '-esJrBWj2d8': 11, 'JKpqYvAdIsw': 25, 'byxOvuiIJV0': 31, 'xxdtq8mxegs': 3, 'EYqVtI9YWJA': 15, 'XkqCExn6_Us': 57, 'oDXZc0tZe04': 13, 'LRw_obCPUt0': 11, 'eQu1rNs0an0': 9, '37rzWOQsNIw': 17, 'EE-bNr36nyA': 5, 'XzYM3PfTM4w': 3, 'sTEELN-vY30': 25, 'WG0MBPpPC6I': 67, 'jcoYJXDG9sw': 5, 'E11zDS9XGzg': 17, '98MoyGZKHXc': 63, 'Se3oxnaPsz0': 43, 'iVt07TCkFM0': 17, 'Bhxk-O1Y7Ho': 35, 'WxtbjNsCQ8A': 25, 'qqR6AEXwxoQ': 5, '91IHQYk1IQM': 59, 'RBCABdttQmI': 63, 'i3wAGJaaktw': 9, 'HT5vyqe0Xaw': 13, '_xMr-HKMfVA': 43, 'vdmoEJ5YbrQ': 23, 'Hl-__g2gn_A': 15, 'akI8YFjEmUw': 9, 'xmEERLqJ2kU': 7, 'uGu_10sucQo': 45, 'Yi4Ij2NM7U4': 9, 'GsAD1KT1xo8': 31, '4wU_LUjG5Ic': 59, 'PJrm840pAUI': 65, 'gzDbaEs1Rlg': 13}
        MN_FRAMES = 150
    elif args.ds == 'summe':
        MN_FRAMES = 150
        og_data = {'Air_Force_One': 2, 'Bearpark_climbing': 27, 'Base jumping': 21, 'Uncut_Evening_Flight': 11, 'Statue of Liberty': 3, 'Eiffel Tower': 21, 'Car_railcrossing': 3, 'paluma_jump': 17, 'Scuba': 25, 'Notre_Dame': 23, 'Paintball': 3, 'Excavators river crossing': 13, 'Kids_playing_in_leaves': 13, 'Bus_in_Rock_Tunnel': 15, 'car_over_camera': 5, 'Cockpit_Landing': 17, 'Bike Polo': 25, 'Jumps': 9, 'Saving dolphines': 9, 'St Maarten Landing': 7, 'Cooking': 21, 'playing_ball': 21, 'Playing_on_water_slide': 17, 'Valparaiso_Downhill': 23, 'Fire Domino': 23}
    else :
        og_data = {}

    device = "cuda"
    my_model = myModel(args)
    my_model.init_pipline_dirs(['sceneDesc', 'FrameEmb','PredMetaData'])

    # init embedder
    embedder = Embedder(device)
    # init Clusterer
    #cluster_algo = ClusterFrame() 


    for video_name in videos_names:
        my_model.set_video_meta_data(video_name)
        if os.path.exists(my_model.prediciton_meta_data_file):
            continue
        print(f'{video_name} Starting ...')
        torch.cuda.empty_cache()

        #scene detection
        scene_list, start_frames = my_model.detect_scences(sst=None)
        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)

        #cache frame embeddings 
        frames = fetch_frames(my_model.video_path)
        if not os.path.exists(my_model.frame_emb_file + '.npy') :
            embedder.cache_frame_embeddings(my_model.frame_emb_file, frames)

        start_frames = my_model.merge_scenes(start_frames, my_model.frame_emb_file + '.npy', min_frames=150)
        
        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)
        print(my_model.video_fps)
        print(my_model.n_frames)
        print(start_frames)
        
        # generate scenes discriptions
        #scene_discription_file_name = my_model.generate_scene_descriptions(frames, start_frames)

        

        
        

    #print(f'TOKEN CNT : {my_model.token_count}')
    return 0

if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="MAIN")
    # Add arguments
    #parser.add_argument("--openai_key", type=str, default='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')
    parser.add_argument("--video_name", type=str, default="")
    parser.add_argument("--ds", type=str,choices=['summe','tvsum', 'other'] )
    parser.add_argument("--video_dir", type=str, help="video/s directory")
    parser.add_argument("--video_type", type=str, choices=['mp4', 'webm'], help='mp4 or webm')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--description_model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--min_scene_duration", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=2)
    args = parser.parse_args()
    run(args)
"""
 python /root/vidSum/src/model/scene_detect.py \
 --video_name EXeTwQWrcwY  \
 --ds other \
 --video_dir /root/data/youtube \
 --video_type mp4 \
 --work_dir /root/YouTube 
"""

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




def masked_scene_prompt(video_description, part_description):
    prompt = ""

    prompt += "You are tasked with evaluating the importance of a specific scene within a larger video. Your goal is to summarize the entire video in just a few sentences, and then assess how much the specific missing scene (replaced by a black screen with 'SCENE MASKED') contributes to the overall summary. Below are two descriptions:\n\n"
    
    prompt += "1. Whole Video Description: This description provides a summary of the entire video, including all scenes. It gives an overall picture of the video's key events, messages, and emotional tone.\n"
    prompt += "2. Masked Scene Video Description: This description provides a summary of the whole video with a specific scene replaced by a black screen displaying the text 'SCENE MASKED.' This means the scene itself is not shown, and only the black screen is visible in place of the scene.\n\n"
    
    prompt += "Your task is to evaluate how critical the missing scene is to a short summary of the video. Imagine that you're summarizing the entire video in a couple of sentences. How much would the absence of this scene affect the completeness of your summary? Does the scene provide essential context, emotional weight, or key details that make it necessary for a well-rounded summary?\n\n"
    
    prompt += "Please assign an importance score on a scale of 1 to 100 based on how much the missing scene contributes to an accurate and complete summary of the video:\n\n"
    
    prompt += "Scale:\n"
    prompt += "1-20: Minimally important (contributes very little to the summary).\n"
    prompt += "21-40: Somewhat important (offers some context or details).\n"
    prompt += "41-60: Moderately important (provides useful context).\n"
    prompt += "61-80: Quite important (adds significant context).\n"
    prompt += "81-100: Highly important (crucial to understanding the video).\n\n"
    
    prompt += "When evaluating, think about how the scene fits into the overall flow of the video and whether its absence leaves a gap in the summary that would affect the viewer's understanding of the main narrative or emotional tone.\n\n"
    
    prompt += "Please provide just a number as the score, with no explanation, and don't add a point at the end.\n\n"
    
    prompt += "Whole Video Description:\n"
    prompt += f"[{video_description}]\n\n"
    
    prompt += "Masked Scene Video Description:\n"
    prompt += f"[{part_description}]"

    return prompt


def masked_scenes_prompt_all(video_description, total_frames, masked_scenes):
    """
    Generates a prompt to evaluate the importance of multiple masked scenes in a video summary,
    incorporating total frame count and per-scene frame counts with start/end boundaries.

    :param video_description: The full description of the video.
    :param total_frames: Total number of frames in the video.
    :param masked_scenes: A list of tuples (scene_number, masked_description, scene_frames, (start_frame, end_frame)).
    :return: A formatted prompt string.
    """

    prompt = (
        "You are tasked with evaluating the importance of specific scenes in a video summary. "
        "Your goal is to assess how much each missing scene (replaced with a black screen labeled 'SCENE MASKED') "
        "affects the completeness of the summary.\n\n"

        f"The **total number of frames** in the original video is **{total_frames}**.\n"
        "Each masked scene's frame count and its exact **start and end positions** are provided. "
        "Keep in mind that the final summary should only be **15% of the original video size**, "
        "so only the most essential scenes should receive high scores.\n\n"

        "**Evaluation Criteria:**\n"
        "- **Whole Video Description**: A summary of the full video, covering all scenes.\n"
        "- **Masked Scene Descriptions**: Summaries where specific scenes have been replaced with a black screen labeled 'SCENE MASKED'.\n\n"

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

        "**Masked Scene Descriptions:**\n"
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

    # generate scenes description
    def generate_scene_descriptions(self, video_frames, scene_frames):

        file_name = self.scene_description_dir + '/' + self.video_name + '.json'
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            return file_name

        scene_descriptions = {}
        all_video_description = self.description_model.generate_description_batch_frames(video_frames, self.video_fps, 1, self.batch_size).replace('\n',' ')

        scene_descriptions['video_description'] = all_video_description
        for i, (start, end) in enumerate(zip(scene_frames[:-1], scene_frames[1:])):
            print(f'Masking : {start} - {end}')
            part_description = self.description_model.generate_description_batch_frames(video_frames, self.video_fps, 1, self.batch_size, mask=(start, end)).replace('\n',' ')
            scene_descriptions[f"scene_{i+1}_description"] = part_description


        with open(file_name, 'w') as json_file:
            json.dump(scene_descriptions, json_file, indent=4)

        return file_name

    # scene score
    def compute_scenes_score(self, discription_file_name):
        # load descriptions
        descriptions = []
        with open(discription_file_name, "r") as json_file:
            descriptions = json.load(json_file)

        scores = []
        for i in range(1,len(descriptions.keys())):
            input_text = masked_scene_prompt(descriptions['video_description'], descriptions[f'scene_{i}_description'])
            input_size = len(input_text.split(' '))

            if(self.token_count  + input_size > self.TPM):
                print('sleep')
                time.sleep(60)
                self.token_count = 0
            response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                {"role": "user", "content": input_text}
            ])
            #
            output_text = response['choices'][0]['message']['content']
            print(output_text)
            scene_score = int(output_text)
            scores.append(scene_score)

            self.token_count += input_size
            

        return scores
    
    #
    def compute_scenes_score_all(self, discription_file_name, scene_frames):
        # load descriptions
        descriptions = []
        with open(discription_file_name, "r") as json_file:
            descriptions = json.load(json_file)

        masked_scenes = []
        for i,(start, end) in enumerate(zip(scene_frames[:-1], scene_frames[1:])):
            key = (i+1, descriptions[f'scene_{i+1}_description'], end-start, (start, end-1))
            masked_scenes.append(key)
        
        prompt = masked_scenes_prompt_all(descriptions['video_description'], self.n_frames, masked_scenes)
        input_size = len(prompt.split(' '))
        if(self.token_count  + input_size > self.TPM):
            print('sleep')
            time.sleep(60)
            self.token_count = 0

        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                {"role": "user", "content": prompt}
            ], temperature=0.2)
            
        output_text = response['choices'][0]['message']['content']
        scores = output_text.split('\n')
        scores = [x.split(':')[1] for x in scores]
        scores = [int(x.strip()) for x in scores]

        return scores

    
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


    def save_results(self, scene_scores, scene_frames, frames_consistency, frames_dissimilarity):
        
        video_prediction_meta_data = {}
        video_prediction_meta_data['video_path'] = self.video_path
        video_prediction_meta_data['video_fps'] = self.video_fps
        video_prediction_meta_data['scene_scores'] = scene_scores
        video_prediction_meta_data['scene_frames'] = scene_frames
        video_prediction_meta_data['n_frames'] = self.n_frames
        video_prediction_meta_data['consistency'] = frames_consistency
        video_prediction_meta_data['dissimilarity'] = frames_dissimilarity
        video_prediction_meta_data['sst'] = self.selected_sst

        with open(self.prediciton_meta_data_file, 'w') as json_file:
            json.dump(video_prediction_meta_data, json_file, indent=4)


#pip install openai==0.28
def run(args):
    openai.api_key = args.openai_key

    if args.video_name == "":
        videos_names = list(os.listdir(args.video_dir))
        videos_names = [name.split('.')[0] for name in videos_names if name.endswith('.' + args.video_type)]
    else :
        videos_names = [args.video_name]

    device = "cuda"
    my_model = myModel(args)
    my_model.init_pipline_dirs(['sceneDesc', 'FrameEmb','PredMetaData'])

    # init embedder
    #embedder = Embedder_dino(device)
    # init Clusterer
    cluster_algo = ClusterFrame() 
    
    
    if args.ds == 'tvsum':
        og_data = {'kLxoNp-UchI': 45, 'AwmHb44_ouw': 17, 'VuWGsYPqAX8': 65, 'NyBmCxDoHJU': 55, '3eYKfiOEJNs': 47, 'fWutDQy1nnY': 21, '0tmA_C6XwfM': 13, 'JgHubY5Vw3Y': 5, 'cjibtmSLxQ4': 37, 'z_6gVvQb2d0': 33, 'b626MiF1ew4': 57, 'J0nA4VgnoCo': 15, 'xwqBXPGE9pQ': 5, '-esJrBWj2d8': 11, 'JKpqYvAdIsw': 25, 'byxOvuiIJV0': 31, 'xxdtq8mxegs': 3, 'EYqVtI9YWJA': 15, 'XkqCExn6_Us': 57, 'oDXZc0tZe04': 13, 'LRw_obCPUt0': 11, 'eQu1rNs0an0': 9, '37rzWOQsNIw': 17, 'EE-bNr36nyA': 5, 'XzYM3PfTM4w': 3, 'sTEELN-vY30': 25, 'WG0MBPpPC6I': 67, 'jcoYJXDG9sw': 5, 'E11zDS9XGzg': 17, '98MoyGZKHXc': 63, 'Se3oxnaPsz0': 43, 'iVt07TCkFM0': 17, 'Bhxk-O1Y7Ho': 35, 'WxtbjNsCQ8A': 25, 'qqR6AEXwxoQ': 5, '91IHQYk1IQM': 59, 'RBCABdttQmI': 63, 'i3wAGJaaktw': 9, 'HT5vyqe0Xaw': 13, '_xMr-HKMfVA': 43, 'vdmoEJ5YbrQ': 23, 'Hl-__g2gn_A': 15, 'akI8YFjEmUw': 9, 'xmEERLqJ2kU': 7, 'uGu_10sucQo': 45, 'Yi4Ij2NM7U4': 9, 'GsAD1KT1xo8': 31, '4wU_LUjG5Ic': 59, 'PJrm840pAUI': 65, 'gzDbaEs1Rlg': 13}
        MN_FRAMES = 250
    else:
        assert args.ds == 'summe', 'Error in --ds arg'
        MN_FRAMES = 150
        og_data = {'Air_Force_One': 2, 'Bearpark_climbing': 27, 'Base jumping': 21, 'Uncut_Evening_Flight': 11, 'Statue of Liberty': 3, 'Eiffel Tower': 21, 'Car_railcrossing': 3, 'paluma_jump': 17, 'Scuba': 25, 'Notre_Dame': 23, 'Paintball': 3, 'Excavators river crossing': 13, 'Kids_playing_in_leaves': 13, 'Bus_in_Rock_Tunnel': 15, 'car_over_camera': 5, 'Cockpit_Landing': 17, 'Bike Polo': 25, 'Jumps': 9, 'Saving dolphines': 9, 'St Maarten Landing': 7, 'Cooking': 21, 'playing_ball': 21, 'Playing_on_water_slide': 17, 'Valparaiso_Downhill': 23, 'Fire Domino': 23}

    print(f'MNF : {MN_FRAMES}')
    for video_name in videos_names:
        my_model.set_video_meta_data(video_name)
        if os.path.exists(my_model.prediciton_meta_data_file):
            continue
        
        print(f'{video_name} Starting ...')
        torch.cuda.empty_cache()

        #scene detection
        sst = og_data[video_name]
        scene_list, start_frames = my_model.detect_scences(sst)
        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)

        #cache frame embeddings 
        frames = fetch_frames(my_model.video_path)
        #if not os.path.exists(my_model.frame_emb_file + '.npy') :
        #    embedder.cache_frame_embeddings(my_model.frame_emb_file, frames)
        
        start_frames = my_model.merge_scenes(start_frames, my_model.frame_emb_file + '.npy', min_frames=MN_FRAMES)
        
        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)
        # generate scenes discriptions
        scene_discription_file_name = my_model.generate_scene_descriptions(frames, start_frames)

        
        # predict scenes score
        scene_scores = my_model.compute_scenes_score(scene_discription_file_name)
        #scene_scores = np.random.randint(1,100, len(start_frames)-1).tolist()
        
        # predoict frames score
        frames_consistency, frames_dissimilarity = my_model.calc_frames_data(cluster_algo, start_frames, len(frames))
        
        # save results
        my_model.save_results(scene_scores, start_frames, frames_consistency, frames_dissimilarity)

        print(f'{video_name} DONE!')


        
        

    #print(f'TOKEN CNT : {my_model.token_count}')
    return 0

if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="MAIN")
    # Add arguments
    parser.add_argument("--openai_key", type=str)
    parser.add_argument("--video_name", type=str, default="")
    parser.add_argument("--ds", type=str,choices=['summe','tvsum'] )
    parser.add_argument("--video_dir", type=str, help="video/s directory")
    parser.add_argument("--video_type", type=str, choices=['mp4', 'webm'], help='mp4 or webm')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--description_model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--min_scene_duration", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=2)
    args = parser.parse_args()
    run(args)

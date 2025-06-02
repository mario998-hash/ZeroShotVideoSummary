
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
from decord import VideoReader, cpu
import numpy as np
import warnings
import cv2
import textwrap
warnings.filterwarnings("ignore")

class DescriptionGenerator:
    
    
    def __init__(self, pretrained, model_name, conv_model_template, device, device_map='auto'):
        self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="float16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
        self.model.eval()
        self.conv_model_template = conv_model_template
        self.device = device
        self.total_frame_num = None
    

    def add_text_overlay(self, frame, text="SCENE MASKED"):
        # Get frame dimensions
        h, w, _ = frame.shape

        # Set font type and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)  # White color for text

        # Create a black rectangle background for the entire frame
        frame[:] = (0, 0, 0)  # Full black frame

        # Calculate the maximum font scale to fit the frame
        font_scale = 1
        max_text_width = w - 20  # Keep some padding around the edges
        max_text_height = h - 20  # Keep some padding around the edges

        # Split the text into multiple lines if it's too long for one line
        lines = textwrap.wrap(text, width=20)  # Wrap text to fit the width

        # Calculate the total height for the wrapped text
        total_text_height = 0
        for line in lines:
            (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, 1)
            total_text_height += text_h + baseline

        # Dynamically adjust font scale to make text fit the frame vertically
        while total_text_height > max_text_height:
            font_scale -= 0.1
            total_text_height = 0
            for line in lines:
                (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, 1)
                total_text_height += text_h + baseline

        # Calculate the starting Y position to center the text vertically
        current_y = (h - total_text_height) // 2

        # Loop over each line of text and draw it on the frame
        for line in lines:
            (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, 1)
            # Calculate the X position to center the text horizontally
            current_x = (w - text_w) // 2
            # Draw the text on the frame
            cv2.putText(frame, line, (current_x, current_y + text_h), font, font_scale, color, 1, cv2.LINE_AA)
            # Move down for the next line
            current_y += text_h + baseline

        return frame


    def load_frames(self, all_frames, acct_fps, wanted_fps, mask):
        """Loads frames, applies masking, and returns sampled frames."""
        
        all_frames = np.array(all_frames)  # Ensure NumPy array

        # Clone frames to prevent modifying the original video
        masked_frames = all_frames.copy()  

        if mask is not None:
            start_frame, end_frame = mask
            for i in range(start_frame, end_frame):
                masked_frames[i] = self.add_text_overlay(masked_frames[i])  # Apply blacked-out overlay

        # Apply sampling
        fps = round(acct_fps / wanted_fps)
        start_sampling = int(acct_fps // 2)
        frame_idx = [i for i in range(start_sampling, len(masked_frames), fps)]
        sampled_frames = masked_frames[frame_idx]

        return sampled_frames


    def aggregate_descriptions(self, text, phrase_to_replace, replacement_phrase, last):
        text.replace(phrase_to_replace,replacement_phrase)
        return text
    

    def generate_description(self, video_path, start_frame, end_frame, segment_duration, fps):
        # Load video and frames
        video, frame_time = self.load_frames(video_path, start_frame, end_frame, fps)
        
        # Preprocess all frames
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
        
        conv_template = self.conv_model_template
        time_instruction = f"The segment lasts for {segment_duration} seconds, and {len(video)} "
        time_instruction += f"frames are uniformly sampled from it. These frames are located at {frame_time}."
        time_instruction += "Please answer the following questions related to this video."
        
        question = DEFAULT_IMAGE_TOKEN + f"Please describe this segment of a video in detail."
        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        cont = self.model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    

    def generate_description_batch_frames(self, video_frames, acct_fps, wanted_fps, batch_size, mask=None):

        video = self.load_frames(video_frames, acct_fps, wanted_fps, mask)
        
        # Preprocess all frames
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
        
        # Split frames into batches
        batched_frames = torch.split(video, batch_size)  # Split video frames into batches
        
        conv_template = self.conv_model_template

        question = DEFAULT_IMAGE_TOKEN + f"Please describe this video in details."
        if mask is not None :
            question += "Please make sure to mention any frames that are masked, scenes that contain the text 'SCENE MASKED' "

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        
        # Step 4: Generate description in batches
        all_descriptions = ""
        
        for batch in batched_frames:
            # Pass each batch of frames to the model
            cont = self.model.generate(
                input_ids,
                images=[batch],  # Process the current batch of frames
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            # Decode and accumulate the results
            text_output = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            all_descriptions += text_output + " "  # Combine results from all batches

        # descriptions aggregation
        phrase_to_replace = "The video begins"# start of the video
        replacement_phrase = "The video continues"
        all_descriptions = self.aggregate_descriptions(all_descriptions, phrase_to_replace, replacement_phrase, last=False)

        phrase_to_replace = "The video ends" # ending of the video
        replacement_phrase = "The video concludes"
        all_descriptions = self.aggregate_descriptions(all_descriptions, phrase_to_replace, replacement_phrase, last=True)
        return all_descriptions.strip()


    def generate_description_batch_frames_v2(self, video_frames, acct_fps, wanted_fps, batch_size, mask=None):

        video = self.load_frames(video_frames, acct_fps, wanted_fps, mask)
        
        conv_template = self.conv_model_template
        question = DEFAULT_IMAGE_TOKEN + f"Please describe this video in details."
        if mask is not None :
            question += "Please make sure to mention any frames that are masked, scenes that contain the text 'SCENE MASKED' "

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        
        # Step 4: Generate description in batches
        n_frames = video.shape[0]
        all_descriptions = ""
        for start_frame in range(0, n_frames, batch_size):
            end_frame = min(start_frame + batch_size, n_frames)
            batch = video[start_frame : end_frame]
            torch_batch = torch.from_numpy(batch)
            video_batch = self.image_processor.preprocess(torch_batch, return_tensors="pt")["pixel_values"].cuda().half()
            # Pass each batch of frames to the model
            cont = self.model.generate(
                input_ids,
                images=[video_batch],  # Process the current batch of frames
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            # Decode and accumulate the results
            text_output = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            all_descriptions += text_output + " "  # Combine results from all batches

        # descriptions aggregation
        phrase_to_replace = "The video begins"# start of the video
        replacement_phrase = "The video continues"
        all_descriptions = self.aggregate_descriptions(all_descriptions, phrase_to_replace, replacement_phrase, last=False)

        phrase_to_replace = "The video ends" # ending of the video
        replacement_phrase = "The video concludes"
        all_descriptions = self.aggregate_descriptions(all_descriptions, phrase_to_replace, replacement_phrase, last=True)
        return all_descriptions.strip()
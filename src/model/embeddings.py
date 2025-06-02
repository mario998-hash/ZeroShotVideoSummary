import os
import numpy as np
import torch
import cv2
# embeddings
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer
from PIL import Image
import torch.nn.functional as F
import json

class Embedder:

    def __init__(self, device, model_name='openai/clip-vit-large-patch14'):

        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.preprocess = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.device = device

    # mean over the segment's frames embeddings
    def get_mean_embedding(self, segment_frames):
        embeddings = []
        for frame in segment_frames:
            inputs = frame.to(self.device)
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)
        return torch.mean(torch.stack(embeddings), dim=0)
    
    # cache video embedding
    def cache_embeddings(self, embedding_file, segments):

        proccessed_segments = []
        for segment in segments:
            proccessed_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in segment]
            proccessed_segment = [self.preprocess(images=frame, return_tensors="pt") for frame in proccessed_frames]
            proccessed_segments.append(proccessed_segment)
    
        segments_embeddings = []
        for segment in proccessed_segments:
            mean_segment_embedding = self.get_mean_embedding(segment)
            segments_embeddings.append(mean_segment_embedding)

        # save embeddings
        embeddings_array = np.vstack([tensor.cpu().numpy() for tensor in segments_embeddings])
        np.save(embedding_file, embeddings_array)


    def cache_frame_embeddings(self, embedding_file, frames):

        proccessed_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        proccessed_frames = [self.preprocess(images=frame, return_tensors="pt") for frame in proccessed_frames]
        embeddings = []
        for frame in proccessed_frames:
            inputs = frame.to(self.device)
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)

        embeddings_array = np.vstack([tensor.cpu().numpy() for tensor in embeddings])
        np.save(embedding_file, embeddings_array)

    
    def cache_frame_embeddings_v2(self, embedding_file, frames, batch_size=16):
        print('cache_frame_embeddings_v2')
        # Convert frames to PIL images
        pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        
        embeddings = []

        for i in range(0, len(pil_frames), batch_size):
            batch_images = pil_frames[i:i + batch_size]

            # Preprocess the batch and send to GPU
            inputs = self.preprocess(images=batch_images, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                batch_embeddings = self.model.get_image_features(**inputs)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            embeddings.append(batch_embeddings.cpu())  # Move to CPU to free GPU

            del inputs, batch_embeddings
            torch.cuda.empty_cache()  # Clear unused memory

        # Stack all batches and save
        embeddings_array = np.vstack([e.numpy() for e in embeddings])
        np.save(embedding_file, embeddings_array)


    def get_query_embedding(self, query: str):
        query = query.strip().lower()
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=1)  # Ensure L2 normalized

        return text_features 
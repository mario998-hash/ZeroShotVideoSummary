import os
import numpy as np
import torch
import cv2
# embeddings
#from transformers import DinoImageProcessor, DinoModel
from PIL import Image
import torch.nn.functional as F
import json
from transformers import AutoProcessor, AutoModel

class Embedder_dino:

    def __init__(self, device, model_name="facebook/dinov2-large" ):

        #self.model = DinoModel.from_pretrained(model_name)
        #self.processor = DinoImageProcessor.from_pretrained(model_name)
        self.preprocess = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
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
                outputs = self.model(**inputs, output_hidden_states=True)
                embedding = outputs.last_hidden_state   # This is a tuple of all hidden states from each layer
                embedding = F.normalize(embedding, p=2, dim=1)
                embedding = embedding.mean(dim=1).squeeze(0) 
                embeddings.append(embedding)

        embeddings_array = np.vstack([tensor.cpu().numpy() for tensor in embeddings])
        np.save(embedding_file, embeddings_array)

    

# Thesis_VideoSummrization

## Description
Zero-Shot Multi-Model Text querable video summarization model \
Input : Video (Frames) \
Ouput : Importance score (Frames)

## Algorithm V1/2
- Scene Detection - Threshold base (mean pixel values segmentation)
- Scene Description generation (LLM based)
- Video/Scene improtance scoring (LLM as a judge)
- Segments extraction - segment is a two second part of contiunes frames (dataset based)
- Segments Description generation (LLM based)
- Scene/segment improtance scoring (LLM as a judge)
- Weighted importance score calc (video/Scene score & Scene/segment score based)
- Hueristice score prediction : Decay + similarity (Clip/Dino based)
- PoR Evaluation (benchmark) 

## Algorithm V4/5
- Scene Detection - Threshold base (mean pixel values segmentation)
- Scene Description generation with masking(LLM based)
- Video/Scene improtance scoring (LLM as a judge)
- Segments extraction - segment is a two second part of contiunes frames (dataset based)
- Segments weight calc - Cluster based
- PoR Evaluation (benchmark)

## Refrences
* Evaluation Paper (PoR) : https://www.iti.gr/~bmezaris/publications/acmmm2020_preprint.pdf 
* Unofficial implementation (github) : https://github.com/e-apostolidis/PoR-Summarization-Measure/blob/master 
* Description generator : We use a version of the LLM family called Qwen 2.0 - Qwen2.0-7B  
  Tutorial @ https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision_Tutorials.ipynb 
* Scoring judege : Currenrly the model is using gpt but there is a free version coming soon (WIP).

## Versions 
* gpt_prediction V1 : Scene and Segment description based - was a bug
* gpt_prediction V2 : Scene and Segment description based - accuracy : 40.41
* gpt_prediction V3 : Leave One Out w.o/ masking - didn't work (No temporal context and relationship between the scene's descriptions)
* gpt_prediction V4 : Leave One Out w/ masking plus embeddings for the "segments" - accuracy with tune : 46.76 
* gpt_prediction V5 : Same as V4 but with different prompt (and scale) - accuracy and tune : WIP 


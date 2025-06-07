# Prompts to Summaries: Zero‑Shot Language‑Guided Video Summarization
This is the offical implementation of the paper : [Prompts to Summaries: Zero‑Shot Language‑Guided Video Summarization](https://arxiv.org/abs/1234.56789)\
![Demo image](./PipeLine_Teaser.jpeg)

---

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)  
- [How to Run](#how-to-run)  
- [License](#license)  
- [Contact](#contact)  

---
## Requirements
Flash-attention
```bash
pip install flash-attn==2.1.0 --no-build-isolation
```

accelarete
```bash
pip install 'accelerate>=0.26.0'
```
libgl1
```bash
apt-get update && apt-get -y install libgl1
```
Video Language Model\
The project uses Qwen2.0 models zoo \
Clone and install :
```bash
git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
```
Large Language Model
```bash
The project uses openAI models so you need a key to use it
```

## Installation
```bash
mkdir vidSum
# Navigate into the directory
cd vidSum

# Clone the repo
git clone https://github.com/mario998-hash/Thesis_VideoSummrization.git

# Install dependencies
pip install -r requirements.txt
```

---
  
## How to Run

### Standerd Vidoe Summarization (SumMe & TVSum datasets)
MetaData
```python
python src/model/solver.py \
--video_name video/name/ \ #or leave it empty if you want to proccess all the video in video_dir 
--video_type /mp4/or/webm \ #video type
--video_dir /path/to/video/directory \
--work_dir /path/to/work/directory \ 
--openai_key /your/openAI/key \ #to use GPT models
```

Evaluation
```python
python src/evaluation/eval.py \
--work_dir /path/to/work/directory \ 
--gt_file /path/to/eccv16_dataset_<summe/tvsum>_google_pool5.h5 \
--splits_file /path/to/Standard-VidSum/splits/<sumMe/tvSum>_splits_5.json
--mapping_file /path/to/Standard-VidSum/splits/<sumMe/tvSum>_mapping.json \
--meta_data_dir /work_dir \
--metric summe/tvsum \
--norm ['None', 'MinMax', 'Exp','MinMax+Exp'] # one of them
```

### Query-Focused Vidoe Summarization (QFVS datasets)
MetaData
```python
python src/model/QFVS_solver.py \
--openai_key /your/openAI/key \ #to use GPT models \
--video_name ['P01','P02','P03','P04'] \ # one of them
--video_dir /path/to/video/directory \
--video_type mp4 \
--work_dir /path/to/work/directory \
--segment_duration 1 \
--mapping_file /path/to/QFVS/QFVS_mapping.json
```
To generate metaData for other Segment durations $W$, run:
```python
pytohn src/model/gpt_tune_QFVS.py \
--video_name ['P01','P02','P03','P04'] \ # one of them
--video_dir /path/to/video/directory \
--video_type mp4 \
--work_dir /path/to/work/directory \
--og_PredMetaData_dir /path/to/MetaDataFile/already/generated \
--segment_duration 2 \ # could be any int (in seconds)
--mapping_file /path/to/QFVS/QFVS_mapping.json
```
Evaluation
```python
python QFVS/evaluation/QFVS_eval.py \
--work_dir /path/to/work/directory \ 
--splits_file /path/to/QFVS/GFVS_splits.py \
--mapping_file /path/to/QFVS/QFVS_mapping.json \
--Tage_file /path/to/QFVS/Tags.mat \
--gt_dir /path/to/Datasets/QFVS/data/origin_data/Query-Focused_Summaries/Oracle_Summaries \
--meta_data_dir /path/to/work/directory \ # same work directory 
--norm ['None', 'MinMax', 'Exp','MinMax+Exp'] # one of them
```

### Fine-Grained Vidoe Summarization (VidSum-Reason datasets)
MetaData \
Need to update the params inside the script
```bash
./VidSum-Reason/MetaData.sh
```

Evaluation
```python
python VidSum-Reason/evaluation/VidSum-Reason_eval.py \
--work_dir /path/to/work/directory \ 
--splits_file /path/to/VidSum-Reason/data/VidSum-Reason_splits_5.json \
--gt_dir /path/to/VidSum-Reason/data/GT \
--meta_data_dir /path/to/work/directory \ # same work directory
--norm ['None', 'MinMax', 'Exp','MinMax+Exp'] # one of them
--fragment_size 3 \
--summary_portion 36 
```



---
## License


---
## Contact
E-mail : mario.bar98@gmail.com

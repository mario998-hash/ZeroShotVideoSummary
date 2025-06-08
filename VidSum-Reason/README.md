# VidSum-Reason Dataset

**VidSum-Reason** is a dataset designed for *user-guided video summarization with reasoning*. Each sample in the dataset is a video paired with a user query and metadata that captures the context for guided summarization.

---

## 📁 Dataset Structure

<pre>
<code>
VidSum-Reason/
├── data/
│   ├── GT/
│   │   ├── vidQry1.json
│   │   ├── vidQry2.json
│   │   ├── vidQry3.json
│   │   ├── ...
│   ├── VidSum-Reason_split_5.json
│   ├── VidSum-Reason_Category_mapping.json
│   ├── VidSum-Reason_mapping.json
</code>
</pre>

---
🎥 **Download Videos**  
The raw video files used in this dataset can be accessed and downloaded from the following Google Drive folder:

[👉 VidSum-Reason Videos on Google Drive](https://drive.google.com/drive/folders/1IfNGfgqbJIcCPBYXUtsNxfSBoHDQ35FQ?usp=sharing)

---

## 📁 Ground truth Structure

<pre>
<code>
GT /
├── data/
├── ...
├── vidQry{i}.json /
│   ├── video_name,
│   ├── user_query,
│   ├── query_category,
│   ├── video_fps,
│   ├── n_frames,
│   ├── gtscore,
├── ...
</code>
</pre>

---

## 🧠 Mapping File: `VidSum-Reason_mapping.json`

The main metadata file, `VidSum-Reason_mapping.json`, contains the mapping between video-query pairs (VidQry) and their associated information.

### 🔑 Example Entry:
```json
{
    "vidQry_1": {
      "video_id": "GIVerZ9mUpU",
      "query": "Filter any scenes that involve violence"
    },
    "vidQry_3": {
    "video_id": "5L4DQfVIcdg",
    "query": "Focus on scenes with the coin machine"
    },
    ...
}
```
---

## 🧠 Category mapping File: `VidSum-Reason_Category_mapping.json`

The main metadata file, `VidSum-Reason_Category_mapping.json`, contains the mapping between every query class and the video-query pairs (VidQry) associated with that calss.

### 🔑 Example Entry:
```json
{
"Reasoning with General Knowledge": {
        "category_keys": [
            "vidQry_1",
            "vidQry_9",
            ...
        ]
    },
    "Reasoning": {
        "category_keys": [
            "vidQry_2",
            "vidQry_4",
            ...
        ]
    },
    ... 
}
```
---

## 🔑 Splits: `VidSum-Reason_split_5.json`

The main splits file, `VidSum-Reason_split_5.json`, contains the mapping of 5 randomly non-overlapping generaated splits for training, testing  and evlautation of video-query pairs (VidQry)

---

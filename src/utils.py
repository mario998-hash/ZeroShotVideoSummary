import cv2

def write_to_file_with_limit(text, file, max_words_per_line = 25):
    words = text.replace("\n","").split()
    for i in range(0, len(words), max_words_per_line):
        line = " ".join(words[i:i+max_words_per_line])
        file.write(line + '\n')

def get_video_FPS(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # cast float fps to the nearset int number
    video_fps = int(fps + 0.5)
    return video_fps

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
def fetch_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames

def read_non_overlapping_segments(video_path, segment_duration, video_fps):
    cap = cv2.VideoCapture(video_path)
    frames_per_segment = int(segment_duration * video_fps)
    segments = []
    while True:
        segment = []
        for _ in range(frames_per_segment):
            ret, frame = cap.read()
            if not ret:
                break
            segment.append(frame)
        if not segment:
            break
        segments.append(segment)
        # Skip to the next segment
        #cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frames_per_segment)
    cap.release()
    return segments, frames_per_segment

def read_non_overlapping_scenes(video_path, video_data):
    cap = cv2.VideoCapture(video_path)
    scenes = []
    scene_num = 0
    start_frames = video_data['scene_frames']
    if video_data['scene_frames'][-1] < video_data['n_frames']:
        start_frames += [video_data['n_frames']]
   
    while True:
        scene = []
        ret, frame = cap.read()
        if not ret:
            break
        for _ in range(start_frames[scene_num], start_frames[scene_num+1]):
            scene.append(frame)
            ret, frame = cap.read()
            if not ret:
                break
        if not scene:
            break
        scenes.append(scene)
        scene_num += 1
    cap.release()
    return scenes
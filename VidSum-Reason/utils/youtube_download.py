import argparse
import os
import yt_dlp 
from urllib.error import HTTPError

def download_video(url, save_dir):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract the video ID from the URL (you can modify it if needed)
    video_id = url.split('v=')[-1]

    # Configure yt-dlp options
    ydl_opts = {
        'outtmpl': os.path.join(save_dir, f'{video_id}.mp4'),  # Save the video as mp4
        'format': 'bestvideo[ext=mp4]',  # Download only the video stream (no audio)
        'quiet': False  # Show download progress
    }

    # Download the video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Downloading video {video_id}...")
            ydl.download([url])
            print(f"Download completed and saved to {save_dir}/{video_id}.mp4")
        except Exception as e:
            print(f"Failed to download video. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download YouTube video from ID.')
    parser.add_argument('--video_url', type=str, required=True, help='YouTube video ID')
    parser.add_argument('--save_dir', type=str, default='/root/data/youtube', help='Directory to save the video')
    args = parser.parse_args()

    download_video(args.video_url, args.save_dir)
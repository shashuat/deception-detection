import sys
import os
import time
from multiprocessing.pool import ThreadPool
import traceback
import tempfile
import shutil
import subprocess
from yt_dlp import YoutubeDL

def time_to_seconds(time_str):
    """Convert time string to seconds"""
    if not time_str or time_str.strip() == '':
        return 0
        
    # Handle simple seconds format
    if time_str.strip().replace('.', '').isdigit():
        return float(time_str)
    
    # Handle MM:SS format
    parts = time_str.strip().split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        # Try to convert directly
        try:
            return float(time_str)
        except ValueError:
            print(f"Warning: Could not parse time '{time_str}', using 0")
            return 0

class VidInfo:
    def __init__(self, yt_id, file_name, start_time, end_time, outdir):
        self.yt_id = yt_id.strip()
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.out_filename = os.path.join(outdir, file_name.strip() + '.mp4')

def download(vidinfo, cookies=None, cookies_from_browser=None):
    """
    Two-stage download process:
    1. First download the complete video to a temporary file using yt-dlp directly
    2. Then use ffmpeg as a subprocess to cut the segment
    
    Args:
        vidinfo: Video information object
        cookies: Path to cookies file (optional)
        cookies_from_browser: Browser name to extract cookies from (optional)
    """
    # Check if the output file already exists
    if os.path.exists(vidinfo.out_filename):
        return f'{vidinfo.yt_id}, SKIPPED (already exists)'
    
    yt_base_url = 'https://www.youtube.com/watch?v='
    yt_url = yt_base_url + vidinfo.yt_id
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_full_video = os.path.join(temp_dir, "full_video.mp4")
    
    try:
        # Stage 1: Download the full video using yt-dlp directly
        ydl_opts = {
            'format': 'mp4[height<=720]/best[ext=mp4]/best',
            'outtmpl': temp_full_video,
            'quiet': False,
            'no_warnings': False,
        }
        
        # Add cookie options if provided
        if cookies:
            ydl_opts['cookiefile'] = cookies
        if cookies_from_browser:
            ydl_opts['cookies_from_browser'] = cookies_from_browser
        
        with YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video {vidinfo.yt_id}")
            ydl.download([yt_url])
        
        # Check if the download was successful
        if not os.path.exists(temp_full_video) or os.path.getsize(temp_full_video) == 0:
            return f'{vidinfo.yt_id}, ERROR (youtube): Download failed or file is empty'
        
        # Stage 2: Cut the segment using ffmpeg as a subprocess
        print(f"Cutting segment from {vidinfo.start_time}s to {vidinfo.end_time}s")
        
        # Prepare ffmpeg command
        cmd = [
            'ffmpeg', '-y', '-i', temp_full_video,
            '-ss', str(vidinfo.start_time),
            '-to', str(vidinfo.end_time),
            '-c:v', 'libx264', '-crf', '18',
            '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k',
            '-r', '25',
            vidinfo.out_filename
        ]
        
        # Execute the command
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        # Check if the segment was created successfully
        if process.returncode != 0:
            error_message = stderr.decode() if stderr else "Unknown ffmpeg error"
            return f'{vidinfo.yt_id}, ERROR (ffmpeg): {error_message}'
        
        if not os.path.exists(vidinfo.out_filename) or os.path.getsize(vidinfo.out_filename) == 0:
            return f'{vidinfo.yt_id}, ERROR (ffmpeg): Output file is empty or missing'
        
        return f'{vidinfo.yt_id}, DONE!'
        
    except Exception as e:
        return_msg = f'{vidinfo.yt_id}, ERROR: {str(e)}'
        traceback.print_exc()
        return return_msg
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

# For direct execution
if __name__ == "__main__":
    import argparse
    
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Download YouTube video segments')
    parser.add_argument('outdir', help='Output directory for downloaded videos')
    parser.add_argument('csv_file', help='CSV file with video information')
    parser.add_argument('--cookies', help='Path to cookies file for YouTube authentication')
    parser.add_argument('--cookies-from-browser', help='Browser name to extract cookies from (chrome, firefox, opera, edge, safari, chromium)')
    
    args = parser.parse_args()
    
    outdir = args.outdir
    csv_file = args.csv_file
    cookies = args.cookies
    cookies_from_browser = args.cookies_from_browser
    
    os.makedirs(outdir, exist_ok=True)
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        lines = [x.split(',') for x in lines]
        # Skip header
        if len(lines) > 0 and not lines[0][0].strip().isalnum():
            lines = lines[1:]
            
        # yt_id, file_name, start_time, end_time
        vidinfos = []
        for line in lines:
            if len(line) >= 4:
                # Use time_to_seconds for time format conversion
                vidinfos.append(VidInfo(
                    line[0], 
                    line[1], 
                    time_to_seconds(line[2]), 
                    time_to_seconds(line[3]), 
                    outdir
                ))
    
    print(f"Found {len(vidinfos)} videos to process")
    
    bad_files = open(f'bad_files_{os.path.basename(outdir)}.txt', 'w')
    
    # Print cookie information if provided
    if cookies:
        print(f"Using cookies file: {cookies}")
    if cookies_from_browser:
        print(f"Extracting cookies from browser: {cookies_from_browser}")
    
    # Use fewer threads to avoid rate limiting
    # Wrap vidinfos with cookies parameters using a lambda function
    results = ThreadPool(3).imap_unordered(
        lambda vi: download(vi, cookies=cookies, cookies_from_browser=cookies_from_browser), 
        vidinfos
    )
    
    cnt = 0
    for r in results:
        cnt += 1
        print(f"{cnt} / {len(vidinfos)} {r}")
        if 'ERROR' in r:
            bad_files.write(r + '\n')
            
    bad_files.close()
# Directories
root_dir = "/Data/dec/data/downloaded"
save_dir = "/Data/dec/data/face_frames/"
import cv2
from tools import FaceAlignmentTools
import os
import numpy as np
import torch
import sys
from pathlib import Path


# Create output directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Initialize face alignment tool
# Use GPU if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tool = FaceAlignmentTools(
    min_face_size=20,  # Lower minimum face size
    device=device
)

# Check if videos directory exists
if not os.path.exists(root_dir):
    print(f"Error: Videos directory '{root_dir}' does not exist!")
    sys.exit(1)

# Get list of video files
video_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

if not video_files:
    print(f"No video files found in '{root_dir}'")
    sys.exit(1)

total_vids = len(video_files)
print(f"Found {total_vids} video files to process")

vid_num = 1

for file_name in video_files:
    full_path = os.path.join(root_dir, file_name)
    
    # Check if file actually exists and is a valid file
    if not os.path.isfile(full_path):
        print(f"Warning: {full_path} is not a valid file. Skipping.")
        continue
        
    # Show file size as a check
    file_size_mb = Path(full_path).stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 0.1:
        print(f"Warning: File {file_name} is too small ({file_size_mb:.2f} MB). Skipping.")
        continue

    num_frames = 0
    save_name = file_name.split('.')[0]
    
    # Create directory for this video's frames
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Processing.... {file_name}\tProgress {vid_num}/{total_vids} ({round(vid_num/total_vids*100, 2)}%)")
    vid_num += 1

    # Try reading with different backends
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_IMAGES, cv2.CAP_DSHOW]:
        try:
            # Open the video with specified backend
            cap = cv2.VideoCapture(full_path, backend)
            
            if not cap.isOpened():
                print(f"Failed to open video with backend {backend}, trying next...")
                continue
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if width == 0 or height == 0 or total_frames == 0:
                print(f"Invalid video dimensions or frame count using backend {backend}")
                cap.release()
                continue
                
            print(f"Video opened successfully with backend {backend}")
            print(f"Dimensions: {width}x{height}, Total frames: {total_frames}, FPS: {fps}")
            break
            
        except Exception as e:
            print(f"Error opening video with backend {backend}: {e}")
            if 'cap' in locals() and cap is not None:
                cap.release()
            continue
    else:
        # If we get here, all backends failed
        print(f"Failed to open video {file_name} with any backend. Skipping.")
        continue

    # If we get here, the video was opened successfully
    frame_idx = 0
    
    # Save a debug frame to check if video is readable
    ret, debug_frame = cap.read()
    if ret:
        debug_path = os.path.join(save_path, "debug_frame.jpg")
        cv2.imwrite(debug_path, debug_frame)
        print(f"Saved debug frame to {debug_path}")
        # Reset to the beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        print("Failed to read first frame for debugging")
        cap.release()
        continue
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Process every 2nd frame to speed up processing
        if frame_idx % 2 != 0:
            continue
            
        # Print progress occasionally
        if frame_idx % 100 == 0 or frame_idx == 1:
            print(f"Processing frame {frame_idx}/{total_frames}")
            
        # Convert to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Try to detect and align face with different settings
        try:
            # First try with normal alignment
            aligned_img = tool.align(rgb_frame)
            
            # If no face found, try with different settings
            if aligned_img is None:
                # Save a sample of frames that don't have faces detected
                if frame_idx % 100 == 0:
                    debug_path = os.path.join(save_path, f"no_face_{frame_idx}.jpg")
                    cv2.imwrite(debug_path, frame)
                    print(f"No face detected in frame {frame_idx}, saved debug image")
                continue
                
            # Convert back to BGR for saving
            aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
            
            # Ensure consistent size
            aligned_img = cv2.resize(aligned_img, (224, 224))
            
            # Save the frame
            num_frames += 1
            s = f"{num_frames:04d}"
            save_image = os.path.join(save_path, f"frame_{s}.jpg")
            
            cv2.imwrite(save_image, aligned_img)
            
            if num_frames % 10 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames, saved {num_frames} faces")
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            # Save problematic frame for debugging
            if frame_idx % 100 == 0:
                error_path = os.path.join(save_path, f"error_{frame_idx}.jpg")
                cv2.imwrite(error_path, frame)
                print(f"Saved error frame to {error_path}")
            continue
    
    # Release resources
    cap.release()
    
    if num_frames > 0:
        print(f"Completed {file_name} - extracted {num_frames} face frames")
    else:
        print(f"Warning: No faces were extracted from {file_name}")

print("All videos processed successfully!")
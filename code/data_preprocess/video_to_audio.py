import os 
from moviepy import VideoFileClip

target_folder = '/Data/dec/data/audio_files/'
bad_videos = '/Data/dec/data/bad_videos.txt'
with open(bad_videos, 'w') as f:
    f.write('')

for f_name in os.listdir('/Data/dec/data/downloaded/'):

    # load video
    f_path = os.path.join('/Data/dec/data/downloaded/',f_name)

    video = VideoFileClip(f_path, audio=True)

    # save audio
    audio = video.audio
    if audio is None:
        with open(bad_videos, 'a') as f:
            f.write(f_name + '\n')
        continue

    save_name = target_folder + f_name.split('.')[0] + '.wav'
    audio.write_audiofile(save_name)

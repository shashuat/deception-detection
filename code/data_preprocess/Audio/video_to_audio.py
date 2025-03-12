import os 
from moviepy import VideoFileClip

target_folder = '/Data/dec/data/audio_files/'

for f_name in os.listdir('/Data/dec/data/downloaded/'):

    # load video
    f_path = os.path.join('/Data/dec/data/downloaded/',f_name)

    video = VideoFileClip(f_path, audio=True)

    # save audio
    audio = video.audio
    save_name = target_folder + f_name.split('.')[0] + '.wav'
    audio.write_audiofile(save_name)

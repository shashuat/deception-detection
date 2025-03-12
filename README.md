## Install Instructions

Download and unzip `DOLOS.zip`

```bash
pip install -r requirements.txt
```

run `YT_video_downloader2.py` to get video chunks in `code/data_preprocess/data/downloaded`

## Dataset Preprocessing

1. Download Video Chunks
run `our_yt_video_downloader.py` to get video chunks 

```bash
python YT_video_downloader2.py /Data/dec/data/downloaded /Data/dec/DOLOS/dolos_timestamps.csv --cookies /Data/dec/code/data_preprocess/utils/youtube.txt
```

We'll store the downloaded files in: `/Data/dec/data/downloaded`. The `--cookies` arg makes the yt_dlp package not think of us as a bot.

I'm not sure if video chopper.py is required because it takes as input long format videos but with yt_video_downloader we already get the 2-5s clips.

2. Extract RGB frames






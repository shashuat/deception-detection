## Install Instructions

Download and unzip `DOLOS.zip`

```bash
pip install -r requirements.txt
```
i did pip freeze > requirements_new.txt (15/03, 12:30 this works)
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

3. Extract Face Frames

Finally, you should have such a data directory structure

```
/Data/dec/data/
├── audio_files/          # .wav audio files 
├── face_frames/          # Extracted face frames as JPG files
│   ├── clip_name1/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── ...
└── rgb_frames/           # Full video frames (optional)
    ├── clip_name1/
    └── ...

/Data/dec/DOLOS/protocols/
├── train_fold1.csv
├── test_fold1.csv
├── train_fold2.csv
└── ...
```

## Running the model
1. Configure the model in `train_test.py` by modifying the `config` dictionary:

```python
config = {
    # Paths
    "data_root": "/path/to/DOLOS/",
    "audio_path": "/path/to/audio_files/",
    "visual_path": "/path/to/face_frames/",
    
    # Model configuration
    "model_to_train": "fusion",  # Options: "audio", "vision", "fusion"
    "num_encoders": 4,
    "adapter": True,
    "adapter_type": "efficient_conv",  # Options: "nlp", "efficient_conv"
    "fusion_type": "cross2",  # Options: "concat", "cross2"
    
    # Training parameters
    ...
}
```

```bash
python -m train_test
```

## Available Models

### Audio-Only Model
Uses Wav2Vec2 with a classification head for audio-only deception detection.

### Visual-Only Model
Uses a CNN feature extractor followed by a Vision Transformer with a classification head.

### Multimodal Fusion Model
Combines audio and visual features using different fusion strategies:
- `concat`: Simple concatenation of features
- `cross2`: Cross-modal attention-based fusion

## Parameter-Efficient Fine-Tuning

The implementation supports two types of adapters:
- `nlp`: Traditional bottleneck adapters
- `efficient_conv`: Convolutional adapters for efficient transfer learning

## Results

Trained models and logs will be saved in the `logs` directory

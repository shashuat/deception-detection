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
## Code directory
```
/Deception-Detection/code
├── archive/          # unused random codes
├── data_preprocess/  # downloading data, frame
├── dataloader/       # contains the torch dataset class
├── models/           # torch model
├── train_test.py     # current training script
├──
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

## Current Warnings
... and 136 more issues
  warnings.warn(f"  - ... and {len(problem_files) - 5} more issues")
Dataset validated: 407 valid samples out of 548 total.
Training samples: 857, Test samples: 407
Model created: DOLOS_fusion_Encoders_4_Adapter_True_type_efficient_conv_fusion_cross2
Starting training for 20 epochs

## Some terminal outputs (just for reference)

Epoch 3/20
Batch 0, Loss: 0.69411
Batch 10, Loss: 0.71668
Batch 20, Loss: 0.66646
Batch 30, Loss: 0.70222
Batch 40, Loss: 0.69655
Batch 50, Loss: 0.68353
Training completed in 86.14 seconds
Epoch 3 Results:
  Train - Loss: 0.68658, Acc: 0.55309, F1: 0.66550, AUC: 0.52947
  Valid - Loss: 0.67329, Acc: 0.59214, F1: 0.67063, AUC: 0.57660
New best model saved to logs/best_model_train_fold3_test_fold3.pt

Epoch 4/20
Batch 0, Loss: 0.67514
Batch 10, Loss: 0.66677
Batch 20, Loss: 0.63368
Batch 30, Loss: 0.75176
Batch 40, Loss: 0.61897
Batch 50, Loss: 0.63325
Training completed in 87.09 seconds
Epoch 4 Results:
  Train - Loss: 0.68696, Acc: 0.57060, F1: 0.63780, AUC: 0.55928
  Valid - Loss: 0.67313, Acc: 0.56265, F1: 0.66160, AUC: 0.54251

Epoch 5/20
Batch 0, Loss: 0.70214
Batch 10, Loss: 0.76371

the initial epoch precision recall values might be NaN, hence a warning is to be expected

## Training FLOPS
gpu usage: python - 22990MiB !!! 22GB damn

## TODO
shashwat:
will implement wandb for train run loggin
+ advancement in architecture backbones

enzo:
will implement different multimodal approach?
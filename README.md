## Install Instructions

Download and unzip `DOLOS.zip`

Download and unzip `data.zip`to avoid to recompute all data processing (~7GB of data)

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
/data/
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

/DOLOS/protocols/
├── train_fold1.csv
├── test_fold1.csv
├── train_fold2.csv
└── ...
```
## Code directory
```
/code
├── archive/          # unused random codes
├── data_preprocess/  # downloading data, frame
├── dataloader/       # contains the torch dataset class
├── models/           # torch model
├── train_test.py     # current training script
├── ...
```

## Running the model
1. Configure the model in `train_test.py` by modifying the `config` dictionary if needed

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

...

Training completed in 83.71 seconds
Epoch 20 Results:
  Train - Loss: 0.63875, Acc: 0.63827, F1: 0.64932, AUC: 0.63988
  Valid - Loss: 0.67102, Acc: 0.60688, F1: 0.70696, AUC: 0.58302

Training completed.
Best Results (Epoch 16) - Acc: 0.61671, F1: 0.64545, AUC: 0.61417

the initial epoch precision recall values might be NaN, hence a warning is to be expected

## Training FLOPS
gpu usage: python - 22990MiB !!! 22GB damn <--- not surprising i think => batches == parallelization (use moge GPU, reduce batch size if out of CUDA memory) /!\ Initial model already do parallelization so can be expensive (but faster)


## TODO
shashwat:
will implement wandb for train run loggin
+ advancement in architecture backbones

enzo:
will implement different multimodal approach
- Start by creating wav to text
- make the code architecture more dynamic / light (will try)
- Create a way to add our own sample from video handmade (web interface to do it ?) --> check how to link web from polytechnics computers to our

both: posters !!!
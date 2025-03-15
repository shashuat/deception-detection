import os
import pandas as pd
from PIL import Image
import numpy as np
import warnings
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchaudio

from utils.console import Style


class DatasetDOLOS(Dataset):
    def __init__(self, 
            annotations_file, audio_dir, img_dir, transcripts_dir,
            num_tokens=64, frame_size=160, ignore_audio_errors=False
        ):
        super(DatasetDOLOS, self).__init__()

        # Load annotations without headers - format: [file_name, label, gender]
        self.annos = pd.read_csv(annotations_file, header=None, names=['file_name', 'label', 'gender'])
        
        # Clean up column data - strip whitespace
        if 'file_name' in self.annos.columns:
            self.annos['file_name'] = self.annos['file_name'].astype(str).str.strip()
        if 'label' in self.annos.columns:
            self.annos['label'] = self.annos['label'].astype(str).str.strip()
        if 'gender' in self.annos.columns:
            self.annos['gender'] = self.annos['gender'].astype(str).str.strip()
            
        self.audio_dir = audio_dir  # all files in '.wav' format
        self.num_tokens = num_tokens
        self.ignore_audio_errors = ignore_audio_errors  # Flag to ignore audio loading errors

        self.img_dir = img_dir
        self.frame_size = frame_size  # Image size (default 160x160)

        self.transcripts_dir = transcripts_dir # all json file from audio transcription (with whisper)
        
        # Use a transform that ensures the output size is always fixed
        self.transforms = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            # Use a fixed-size resize that maintains aspect ratio and then center crops
            T.Resize(frame_size),
            T.CenterCrop(frame_size),  # This ensures all images are exactly frame_size x frame_size
            # normalize to imagenet mean and std values
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.number_of_target_frames = num_tokens
        
        # Filter out samples with missing or invalid data
        self._validate_samples()

    def _validate_samples(self):
        """Validate all samples and filter out problematic ones"""
        valid_indices = []
        problem_files = []
        
        print("Validating dataset samples...")
        for idx, row in self.annos.iterrows():
            clip_name = row['file_name']
            
            # Check for valid label
            label_str = str(row['label']).strip().lower()
            if label_str not in ['truth', 'truthful', 'deception', 'lie', '0', '1']:
                problem_files.append(f"Invalid label: '{row['label']}' for clip {clip_name}")
                continue
                
            # Check if audio file exists
            audio_path = os.path.join(self.audio_dir, clip_name + '.wav')
            if not os.path.exists(audio_path):
                problem_files.append(f"Missing audio file: {audio_path}")
                continue
                
            # Check if frames directory exists
            frames_dir = os.path.join(self.img_dir, clip_name)
            if not os.path.exists(frames_dir):
                problem_files.append(f"Missing frames directory: {frames_dir}")
                continue
                
            # Check if frames exist
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg') and f != 'debug_frame.jpg']
            if len(frame_files) == 0:
                problem_files.append(f"No frame files in: {frames_dir}")
                continue

            # Check if transcript exist
            transcript_file = os.path.join(self.transcripts_dir, clip_name + ".json")
            if not os.path.exists(transcript_file):
                problem_files.append(f"Missing transcript file: {transcript_file}")
                continue
                
            # If all checks pass, add to valid indices
            valid_indices.append(idx)
        
        # Print summary of validation
        if problem_files:
            txt = Style("WARNING", f"Found {len(problem_files)} problematic files out of {len(self.annos)}.") + "\n"
            for i, problem in enumerate(problem_files[:5]):  # Show first 5 issues
                txt += f"  - {Style('WARNING', problem)} \n"

            if len(problem_files) > 5:
                txt += f"  ... and {len(problem_files) - 5} more issues\n"

            warnings.warn(txt, stacklevel=2)
            
        
        # Filter annotations to only include valid samples
        self.annos = self.annos.iloc[valid_indices].reset_index(drop=True)
        print(f"Dataset validated: {len(self.annos)} valid samples out of {len(self.annos) + len(problem_files)} total.")

    def _load_audio(self, audio_path):
        """Load and process audio with robust error handling"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Use mono audio (left channel)
            waveform = waveform[0]
            
            # Calculate duration of the audio clip
            clip_duration = len(waveform) / sample_rate
            # For wav2vec2, 1 Token corresponds to ~321.89 discrete samples
            # To get precisely num_tokens, calculate the required sample rate
            new_sample_rate = int(321.893491124260 * self.num_tokens / clip_duration)
            # Resample audio
            waveform = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)
            # Prepare for batch collation
            mono_waveform = waveform.unsqueeze(0)
            mono_waveform = mono_waveform.type(torch.float32)
            
            return mono_waveform
            
        except Exception as e:
            if self.ignore_audio_errors:
                # Return zero waveform of appropriate size
                warnings.warn(f"Error loading audio file {audio_path}: {str(e)}. Using zero waveform instead.")
                # Create fake waveform with approximately the right number of samples
                fake_waveform = torch.zeros(1, int(self.num_tokens * 321.9))
                return fake_waveform
            else:
                # Re-raise with more context
                raise RuntimeError(f"Error loading audio file {audio_path}: {str(e)}") from e

    def _load_frames(self, frames_dir):
        """Load and process face frames with robust error handling"""
        try:
            # Get a list of all jpg files EXCEPT debug_frame.jpg
            frame_names = []
            for f in os.listdir(frames_dir):
                if f.endswith('.jpg') and f != 'debug_frame.jpg':
                    frame_names.append(f)
            
            if not frame_names:
                raise ValueError(f"No valid frame files found in {frames_dir}")
                
            # Sort frames by name to ensure consistent ordering
            frame_names.sort()
                
            # Sample target_frames number of face frames from the available ones
            target_frames = np.linspace(0, len(frame_names) - 1, num=self.number_of_target_frames)
            target_frames = np.around(target_frames).astype(int)
            
            face_frames = []
            for i in target_frames:
                if i >= len(frame_names):  # Handle edge case
                    i = len(frame_names) - 1
                frame_name = frame_names[i]
                
                try:
                    # Load and process the image
                    img_path = os.path.join(frames_dir, frame_name)
                    img = Image.open(img_path)
                    
                    # Apply our transforms to get consistent size
                    transformed_img = self.transforms(img)
                    
                    # Verify the shape is correct
                    if transformed_img.shape[1] != self.frame_size or transformed_img.shape[2] != self.frame_size:
                        warnings.warn(f"Frame size mismatch after transform: {transformed_img.shape} for {img_path}")
                        # Skip this frame if size is wrong
                        continue
                        
                    face_frames.append(transformed_img)
                    
                except Exception as e:
                    warnings.warn(f"Error processing frame {frame_name} in {frames_dir}: {str(e)}")
                    # Skip this frame and continue
                    continue
            
            # Make sure we have enough frames
            if len(face_frames) < self.number_of_target_frames:
                # If we don't have enough frames, duplicate the last one
                warnings.warn(f"Not enough valid frames in {frames_dir}: {len(face_frames)}/{self.number_of_target_frames}")
                
                if len(face_frames) == 0:
                    # Create a black frame if we have none
                    black_frame = torch.zeros(3, self.frame_size, self.frame_size)
                    face_frames = [black_frame] * self.number_of_target_frames
                else:
                    # Duplicate the last frame to reach the required number
                    last_frame = face_frames[-1]
                    while len(face_frames) < self.number_of_target_frames:
                        face_frames.append(last_frame)
            
            # Stack frames and ensure float32 type
            face_frames = torch.stack(face_frames, 0)
            face_frames = face_frames.type(torch.float32)
            
            return face_frames
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Error loading frames from {frames_dir}: {str(e)}") from e
        
    def _load_transcripts(self, transcript_path, size=256):
        transcript = json.load(open(transcript_path, 'r'))
        bert_embedding = torch.tensor(transcript["bert_embedding"]).t() # (256, 768)

        whisper_tokens = []
        for segment in transcript["segments"]:
            whisper_tokens.extend(segment["tokens"])

        if len(whisper_tokens) > size: 
            print("warning - whisper tokens truncated")
            whisper_tokens = whisper_tokens[:size]

        while len(whisper_tokens) < size: # right padding
            whisper_tokens.append(0)
        
        whisper_tokens = torch.tensor(whisper_tokens).t() # (256, )
        return whisper_tokens, bert_embedding
        

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        # Get clip name from annotations
        clip_name = self.annos.iloc[idx, 0]

        # Process audio
        audio_path = os.path.join(self.audio_dir, clip_name + '.wav')
        mono_waveform = self._load_audio(audio_path)

        # Process face frames
        frames_dir = os.path.join(self.img_dir, clip_name)
        face_frames = self._load_frames(frames_dir)

        # Process transcript
        transcript_path = os.path.join(self.transcripts_dir, clip_name + '.json')
        transcript = self._load_transcripts(transcript_path)

        # Process label - make case-insensitive and strip whitespace
        str_label = self.annos.iloc[idx, 1]
        
        # Make sure it's a string, strip whitespace, and convert to lowercase
        if not isinstance(str_label, (int, float)):
            str_label_clean = str(str_label).strip().lower()
        else:
            str_label_clean = str_label
            
        # Match against cleaned versions
        if str_label_clean in ['truth', 'truthful', '0', 0]:
            label = 0
        elif str_label_clean in ['deception', 'lie', '1', 1]:
            label = 1
        else:
            # Print full details about the problematic label including any whitespace
            label_repr = repr(str_label)  # Shows whitespace characters
            raise ValueError(
                f"Undefined label: {label_repr} (type: {type(str_label)}), " 
                f"cleaned: '{str_label_clean}', clip_name: {clip_name}"
            )

        return mono_waveform, face_frames, transcript, label


def af_pad_sequence(batch):
    """Pad audio sequences to the same length"""
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def af_collate_fn(batch):
    """Collate function for batching audio, visual and textual data"""
    tensors, face_tensors, whisper_tensors, bert_tensors, targets = [], [], [], [], []

    # Gather data and encode labels
    for waveform, face_frames, (whisper_tokens, bert_embedding), label in batch:
        tensors.append(waveform)
        face_tensors.append(face_frames)
        whisper_tensors.append(whisper_tokens)
        bert_tensors.append(bert_embedding)
        targets.append(torch.tensor(label))

    # Group the tensors into batched tensors
    tensors = af_pad_sequence(tensors)
    face_tensors = torch.stack(face_tensors)
    whisper_tensors = torch.stack(whisper_tensors)
    bert_tensors = torch.stack(bert_tensors)
    targets = torch.stack(targets)

    return tensors, face_tensors, (whisper_tensors, bert_tensors), targets
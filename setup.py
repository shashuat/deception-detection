import os
import glob
import importlib.util

import sys
import os

try:
    from code.utils.console import Style  # Normal import (when running from project root)
except ImportError:
    # # if import fails, adjust sys.path and retry
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # if project_root not in sys.path:
    #     sys.path.append(project_root)

    from utils.console import Style  # type: ignore

DEFAULT_CONFIG = {
    "DATA_PATH": os.path.join("data"),
    "DOLOS_PATH": os.path.join("DOLOS"),
    "LOGS_PATH": os.path.join("logs"),
}

CONFIG = DEFAULT_CONFIG
config_path = os.path.join(os.path.dirname(__file__), "config.py")

if os.path.exists(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Override defaults with values from config.py (if they exist)
    CONFIG.update({k: getattr(config_module, k, v) or v for k, v in DEFAULT_CONFIG.items()})

def get_path_diff(reference_file="setup.py"):
    """Compute the relative path difference between the script's location and a reference file (e.g., 'setup.py')."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Where the script is located
    project_root = os.path.dirname(os.path.abspath(reference_file))  # Where setup.py is

    # Compute relative path from script to project root
    relative_path = os.path.relpath(script_dir, project_root)

    return relative_path

def get_env(_check_env=False):
    """
        Get paths of the project for the current installation

        Params:
        -----------
        _check_env: bool (False): auto check the env before getting it

        Return:
        -----------
        CONFIG: dict

        {
            "DATA_PATH": ...,
            "DOLOS_PATH": ...,
            "LOGS_PATH": ...,
            "AUDIO_PATH": ...,
            "VISUAL_PATH": ...
        }
    """

    if _check_env: check_env()
    
    relative_path = get_path_diff()

    return_conf = CONFIG.copy()
    return_conf["AUDIO_PATH"] = os.path.join(CONFIG["DATA_PATH"], "audio_files")
    return_conf["VISUAL_PATH"] = os.path.join(CONFIG["DATA_PATH"], "face_frames")
    return_conf["TRANSCRIPTS_PATH"] = os.path.join(CONFIG["DATA_PATH"], "transcripts")

    for k, v in return_conf.items():
        return_conf[k] = os.path.join(relative_path, v) if v[0] != "/" else v
    return return_conf

def check_env(_print=True):
    """
    Check if everything is correctly downloaded.
    
    Steps:
      1. Verify that the main directories (data and DOLOS) and required subdirectories exist.
      2. Check that the static files in the main repositories (audio, face frames, protocols) are present.
      3. Determine if the environment is "pre-initialized" (i.e. critical data are available) and print info with counts.
    """
    if _print and os.path.exists(config_path):
        print(Style("INFO", "Importing configuration from /config.py"))
    elif _print:
        print(Style("INFO", "No custom configuration. Using default one"))

    errors = []
    warnings = []
    info = {}

    # Use CONFIG for main directories
    data_dir = CONFIG["DATA_PATH"]
    dolores_dir = CONFIG["DOLOS_PATH"]

    if not os.path.exists(data_dir):
        errors.append(Style("ERROR", f"Directory '{data_dir}' is missing. You can set a custom path in /config.py (see README)"))
    
    if not os.path.exists(dolores_dir):
        errors.append(Style("ERROR", f"Directory '{dolores_dir}' is missing. You can set a custom path in /config.py (see README)"))

    if errors:
        txt = Style("ERROR", "Environment check failed:") + "\n"
        for e in errors:
            txt += "- " + str(e)

        if not _print:
            raise FileNotFoundError(txt)
        else:
            print(txt)
        return False

    # Check required subdirectories in the data folder
    audio_dir = os.path.join(data_dir, "audio_files")
    face_frames_dir = os.path.join(data_dir, "face_frames")
    rgb_frames_dir = os.path.join(data_dir, "rgb_frames")

    if not os.path.exists(audio_dir):
        errors.append(Style("ERROR", f"Directory '{audio_dir}' is missing."))
    if not os.path.exists(face_frames_dir):
        errors.append(Style("ERROR", f"Directory '{face_frames_dir}' is missing."))
    if not os.path.exists(rgb_frames_dir):
        warnings.append(Style("WARNING", f"Optional directory '{rgb_frames_dir}' is missing."))

    protocols_dir = os.path.join(dolores_dir, "protocols")
    if not os.path.exists(protocols_dir):
        errors.append(Style("ERROR", f"Directory '{protocols_dir}' is missing."))

    if errors:
        print(Style("ERROR", "Environment check failed:"))
        for e in errors:
            print("Error:", e)
        return False

    # Check for static files in 'audio_files'
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    info["audio_files_count"] = len(audio_files)
    if not audio_files:
        warnings.append(Style("WARNING", f"No audio files found in '{audio_dir}'."))

    # Check for static files in 'face_frames'
    total_face_frames = 0
    face_clips = [d for d in os.listdir(face_frames_dir) if os.path.isdir(os.path.join(face_frames_dir, d))]
    info["face_clip_count"] = len(face_clips)
    for clip in face_clips:
        jpg_files = glob.glob(os.path.join(face_frames_dir, clip, "*.jpg"))
        total_face_frames += len(jpg_files)
    info["face_frames_count"] = total_face_frames
    if total_face_frames == 0:
        warnings.append(Style("WARNING", f"No face frame images found in '{face_frames_dir}'."))

    # Optionally check 'rgb_frames'
    total_rgb_frames = 0
    if os.path.exists(rgb_frames_dir):
        rgb_clips = [d for d in os.listdir(rgb_frames_dir) if os.path.isdir(os.path.join(rgb_frames_dir, d))]
        info["rgb_clip_count"] = len(rgb_clips)
        for clip in rgb_clips:
            jpg_files = glob.glob(os.path.join(rgb_frames_dir, clip, "*.jpg"))
            total_rgb_frames += len(jpg_files)
        info["rgb_frames_count"] = total_rgb_frames
    else:
        info["rgb_frames_count"] = 0

    # Check for protocol CSV files in 'DOLOS/protocols'
    protocol_files = glob.glob(os.path.join(protocols_dir, "*.csv"))
    info["protocol_files_count"] = len(protocol_files)
    if not protocol_files:
        warnings.append(Style("WARNING", f"No protocol CSV files found in '{protocols_dir}'."))

    # Determine if the environment is pre-initialized (has all critical data)
    pre_init = (len(audio_files) > 0 and total_face_frames > 0 and len(protocol_files) > 0)
    info["pre_init"] = pre_init

    if pre_init:
        print(Style("SUCCESS", "Environment is properly initialized:"))
        for key, value in info.items():
            print(f"- {key}: {value}")
    else:
        warnings.append(Style("WARNING", "Environment is not fully pre-initialized. Some data is missing."))

    if warnings:
        print(Style("WARNING", "Warnings:"))
        for w in warnings:
            print("- :", w)

    return pre_init

# --------------------------
# RUN FROM project root
# --------------------------
if __name__ == "__main__":
    check_env()

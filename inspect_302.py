import pandas as pd
import os
import numpy as np

DATA_ROOT = r"d:/Dropbox/Collaboration/Prof Hafiz Farooq/iSMART Paper/Final version/code/daic_woz_data"
participant_id = "302"
session_dir = f"{DATA_ROOT}/sessions/{participant_id}_P"

print(f"Checking data for {participant_id}_P...")

# Check COVAREP (Audio)
audio_file = os.path.join(session_dir, f"{participant_id}_COVAREP.csv")
if os.path.exists(audio_file):
    df_audio = pd.read_csv(audio_file, header=None)
    print(f"COVAREP frames: {len(df_audio)}")
else:
    print("COVAREP file missing")

# Check Video
video_file = os.path.join(session_dir, f"{participant_id}_CLNF_features.txt")
if os.path.exists(video_file):
    # CLNF features usually have a header
    df_video = pd.read_csv(video_file)
    print(f"CLNF frames: {len(df_video)}")
else:
    print("CLNF file missing")

# Check Transcript
transcript_file = os.path.join(session_dir, f"{participant_id}_TRANSCRIPT.csv")
if os.path.exists(transcript_file):
    df_text = pd.read_csv(transcript_file, sep='\t')
    print(f"Transcript lines: {len(df_text)}")
    participant_lines = df_text[df_text['speaker'] == 'Participant']
    print(f"Participant lines: {len(participant_lines)}")
else:
    print("Transcript file missing")

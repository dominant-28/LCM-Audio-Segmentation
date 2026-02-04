import os
import torch
import soundfile as sf
import torchaudio
import numpy as np
from pathlib import Path
from wtpsplit import SaT
from faster_whisper import WhisperModel

# Patch temporary directory to prevent system lockups on WSL/Colab
local_tmp = os.path.join(os.getcwd(), "local_tmp")
os.makedirs(local_tmp, exist_ok=True)
os.environ["TMPDIR"] = local_tmp
os.environ["TEMP"] = local_tmp
os.environ["TMP"] = local_tmp

class DatasetProcessor:
    """
    Handles ASR-guided audio segmentation using LCM logic (Whisper + SaT).
    Designed to be robust against dependency conflicts and filesystem errors.
    """
    def __init__(self, output_root="dataset_clean", model_cache_dir="./model_cache"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model_cache = model_cache_dir
        
        # Device selection logic
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
            print(f"Initializing models on GPU ({self.compute_type})...")
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            print(f"GPU not found. Falling back to CPU ({self.compute_type}).")

        # Load Whisper
        os.makedirs(self.model_cache, exist_ok=True)
        self.whisper = WhisperModel(
            "large-v3", 
            device=self.device, 
            compute_type=self.compute_type, 
            download_root=self.model_cache
        )

        # Load SaT (Segmentation as Tokenization)
        # We initialize the wrapper first, then move internals to device to avoid attribute errors
        self.sat = SaT("sat-3l-sm")
        if self.device == "cuda":
            self.sat.half()
            self.sat.to(self.device)
        else:
            self.sat.to(self.device)

    def _apply_lcm_logic(self, text, threshold=0.2, min_len=10, max_len=256):
        """
        Applies the 'Goldilocks' segmentation logic from the Large Concept Model paper.
        1. Split at SaT probability > 0.2
        2. Merge segments shorter than min_len
        3. Force split segments longer than max_len
        """
        sat_splits = self.sat.split(text, threshold=threshold)
        final_segments = []
        buffer_seg = ""

        for seg in sat_splits:
            seg = seg.strip()
            if not seg: continue

            # Merge short segments
            if len(seg) < min_len:
                buffer_seg += " " + seg if buffer_seg else seg
                continue
            
            # Flush buffer
            if buffer_seg:
                if len(buffer_seg) + len(seg) < max_len:
                    seg = buffer_seg + " " + seg
                else:
                    final_segments.append(buffer_seg)
                buffer_seg = ""

            # Handle long segments (Safety Valve)
            if len(seg) > max_len:
                subs = self.sat.split(seg, threshold=0.01) # Aggressive split
                final_segments.extend(subs)
            else:
                final_segments.append(seg)
        
        if buffer_seg: final_segments.append(buffer_seg)
        return final_segments

    def process_file(self, audio_path):
        """
        Processes a single audio file: Transcribe -> Segment -> Align -> Save.
        """
        audio_path = Path(audio_path)
        file_id = audio_path.stem
        
        specific_output_dir = self.output_root / file_id
        if specific_output_dir.exists():
            print(f"Skipping {file_id} (Output directory exists).")
            return

        print(f"Processing: {audio_path.name}")
        specific_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Transcribe with timestamps
        segments, _ = self.whisper.transcribe(str(audio_path), word_timestamps=True)
        
        all_words = []
        full_text_list = []
        for s in segments:
            full_text_list.append(s.text)
            all_words.extend(s.words)
        full_text = "".join(full_text_list)

        # 2. Get Semantic Segments
        semantic_segments = self._apply_lcm_logic(full_text)

        # 3. Load Audio
        try:
            audio_np, sr = sf.read(str(audio_path))
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return

        # Prepare for slicing (Standardize to 16kHz mono)
        wav = torch.from_numpy(audio_np).float()
        if wav.ndim == 2: wav = wav.t()
        else: wav = wav.unsqueeze(0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000
        
        wav = wav.squeeze()
        
        # 4. Alignment Loop
        word_idx = 0
        metadata = []
        
        for i, seg_text in enumerate(semantic_segments):
            seg_clean = seg_text.replace(".", "").lower().strip()
            current_build = ""
            start_t, end_t = None, None

            # Simple greedy alignment
            if word_idx < len(all_words): start_t = all_words[word_idx].start

            while word_idx < len(all_words):
                w = all_words[word_idx]
                current_build += w.word.replace(".", "").lower().strip()
                end_t = w.end
                word_idx += 1
                # Check for approximate match
                if len(current_build) >= len(seg_clean) * 0.9: 
                    break
            
            if start_t is not None and end_t is not None:
                start_sample = int(start_t * sr)
                end_sample = int(end_t * sr)
                
                # Skip segments shorter than 0.1s
                if end_sample - start_sample < 1600: continue

                clip_name = f"seg_{i:04d}.wav"
                clip_path = specific_output_dir / clip_name
                
                # Save clip
                audio_slice = wav[start_sample:end_sample]
                sf.write(str(clip_path), audio_slice.numpy(), sr)
                
                metadata.append(f"{clip_name}|{seg_text}")

        # Save metadata
        with open(specific_output_dir / "metadata.txt", "w") as f:
            f.write("\n".join(metadata))

        print(f"Saved {len(metadata)} clips to {specific_output_dir}")

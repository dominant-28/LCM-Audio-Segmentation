import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.segmentation_pipeline import DatasetProcessor

def main():
    # Configuration
    input_folder = "/content/your_source_audio"  # Update this path
    output_folder = "/content/processed_dataset" # Update this path
    
    # Initialize Pipeline
    processor = DatasetProcessor(output_root=output_folder)
    
    # Collect Files
    input_path = Path(input_folder)
    audio_extensions = {".wav", ".flac", ".mp3"}
    files = [f for f in input_path.iterdir() if f.suffix.lower() in audio_extensions]
    
    print(f"Found {len(files)} audio files in {input_folder}")
    
    # Process
    for f in tqdm(files, desc="Processing Dataset"):
        processor.process_file(f)

if __name__ == "__main__":
    main()

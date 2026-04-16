import json
import argparse
import multiprocessing as mp
import os
from pathlib import Path
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip
from functools import partial


def extract_audio_sample(sample, output_audio_dir, base_dir=None):
    """Extract audio from video file and save as wav."""
    try:
        file_path_str = sample.get('file_path')
        if not file_path_str:
            return sample
        
        # Handle path resolution
        file_path_obj = Path(file_path_str)
        
        # If base_dir is provided and the path is relative, join them
        if base_dir and not file_path_obj.is_absolute():
            file_path_obj = Path(base_dir) / file_path_obj
        
        # Resolve to absolute path to ensure consistency
        final_file_path = str(file_path_obj.resolve())

        if not os.path.exists(final_file_path):
            print(f"Warning: Video file not exists: {final_file_path}")
            return sample
        
        # Check if file is video
        ext = Path(final_file_path).suffix.lower()
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        if ext not in video_extensions:
            print(f"Warning: Not a video file '{ext}' for {final_file_path}")
            return sample
        
        # Extract audio
        # Note: MoviePy might still print some FFmpeg logs to stderr despite verbose=False in write_audiofile
        video = VideoFileClip(final_file_path)
        
        # Construct output wav path
        base_name = os.path.basename(final_file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_audio_path = os.path.join(output_audio_dir, f"{name_without_ext}.wav")
        
        # Save audio as wav
        if video.audio is not None:
            # Ensure the directory for output exists (though main func creates the root dir)
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            
            video.audio.write_audiofile(
                output_audio_path, 
                codec='pcm_s16le', 
                fps=16000, 
                logger=None
            )
            # Add audio_path to sample
            # We save the absolute path or the relative path depending on your needs. 
            # Here we save the generated absolute path for clarity.
            sample['audio_path'] = output_audio_path
        else:
            print(f"Warning: No audio track in {final_file_path}")
        
        # Release resources
        video.close()
        
        return sample
    except Exception as e:
        print(f"Error processing {sample.get('file_path')}: {e}")
        return sample


def process_json_extract_wav(input_json_path, output_json_path, output_audio_dir, base_dir=None, num_processes=None):
    # Create output audio directory
    os.makedirs(output_audio_dir, exist_ok=True)
    
    # Load the JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract samples based on data structure
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    else:
        samples = [data]
    
    # Set number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Starting processing {len(samples)} samples using {num_processes} processes...")
    if base_dir:
        print(f"Using base directory for relative video paths: {base_dir}")
    
    # Process samples using multiprocessing
    with mp.Pool(processes=num_processes) as pool:
        # Use partial to create a pickable function with fixed arguments
        process_func = partial(
            extract_audio_sample, 
            output_audio_dir=output_audio_dir,
            base_dir=base_dir
        )
        processed_samples = pool.map(process_func, samples)
    
    # Reconstruct output data preserving original structure
    if isinstance(data, list):
        output_data = processed_samples
    elif isinstance(data, dict) and 'samples' in data:
        output_data = data.copy()
        output_data['samples'] = processed_samples
    else:
        output_data = processed_samples[0]
    
    print(f"Successfully processed {len(processed_samples)} samples.")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete! Results saved to {output_json_path}")


if __name__ == '__main__':
    # Use 'spawn' start method to avoid potential deadlocks with moviepy in multiprocessing
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Extract audio from videos and add audio_path to JSON metadata."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="datasets/X-Fun-Videos-Audios-Demo/metadata_origin.json",
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="datasets/X-Fun-Videos-Audios-Demo/metadata.json",
        help="Path to the output JSON file."
    )
    parser.add_argument(
        "--output_audio_dir",
        type=str,
        default="datasets/X-Fun-Videos-Audios-Demo/wav",
        help="Directory to save extracted wav files."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory to prepend to relative video file paths in JSON. If not provided, paths are treated as absolute or relative to CWD."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of parallel processes to use. Defaults to CPU core count."
    )
    
    args = parser.parse_args()
    
    process_json_extract_wav(
        input_json_path=args.input_file,
        output_json_path=args.output_file,
        output_audio_dir=args.output_audio_dir,
        base_dir=args.base_dir,
        num_processes=args.num_processes
    )

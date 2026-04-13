import json
import argparse
import multiprocessing as mp
from pathlib import Path
from PIL import Image

# Supported file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}


def process_media_sample(sample):
    """Extract width and height from image or video files."""
    try:
        file_path = sample.get('file_path')
        if not file_path:
            return sample

        ext = Path(file_path).suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            # Extract dimensions from image
            with Image.open(file_path) as img:
                width, height = img.size
        elif ext in VIDEO_EXTENSIONS:
            # Extract dimensions from video using decord
            import cv2
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Warning: Cannot open video {file_path}")
                return sample
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            print(f"Warning: Unsupported format '{ext}' for {file_path}")
            return sample

        # Update sample with extracted dimensions
        sample['height'] = height
        sample['width'] = width
        return sample
    except Exception as e:
        print(f"Error processing {sample.get('file_path')}: {e}")
        return sample


def process_json_with_multiprocessing(input_json_path, output_json_path, num_processes=None):
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

    # Process samples using multiprocessing
    with mp.Pool(processes=num_processes) as pool:
        processed_samples = pool.map(process_media_sample, samples)

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
    # Use 'spawn' start method to avoid potential deadlocks with decord/FFmpeg in multiprocessing
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Add width and height fields to image/video metadata in JSON files."
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="datasets/X-Fun-Images-Demo/metadata.json",
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="datasets/X-Fun-Images-Demo/metadata_add_width_height.json",
        help="Path to the output JSON file."
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=None,
        help="Number of parallel processes to use. Defaults to CPU core count."
    )

    args = parser.parse_args()

    process_json_with_multiprocessing(
        input_json_path=args.input_file,
        output_json_path=args.output_file,
        num_processes=args.num_processes
    )

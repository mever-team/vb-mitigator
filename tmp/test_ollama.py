import os
import subprocess
import sys
import argparse
from tqdm import tqdm
import torch

def get_image_paths(directory):
    exts = ('.png')
    image_paths = [os.path.join(root, f)
                   for root, _, files in os.walk(directory)
                   for f in files if f.lower().endswith(exts)]
    return sorted(image_paths)

def caption_image(image_path, model, prompt, output_file):
    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path}")
        return None

    command = ["ollama", "run", model, image_path, prompt]
    print(f"[DEBUG] Running command: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=1200)
        caption = result.stdout.strip()
        print(f"[INFO] Caption generated for {os.path.basename(image_path)}")

        if output_file:
            with open(output_file, 'a') as f:
                f.write(f"Image: {image_path}\nCaption: {caption}\n{'-'*40}\n")
        return caption

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Timeout expired for image {image_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ollama subprocess error: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

    return None

def main():
    parser = argparse.ArgumentParser(description="Caption images with Ollama (GPU-aware).")
    parser.add_argument("--model", default="llava", help="Ollama model name (e.g. 'llava', 'llava:13b')")
    parser.add_argument("--image_dir", required=True, help="Directory with images")
    parser.add_argument("--prompt", default="Describe this image concisely.", help="Caption prompt")
    parser.add_argument("--output_file", default="image_captions.txt", help="File to save captions")
    args = parser.parse_args()

    #GPU availability
    if torch.cuda.is_available():
        print(f"[INFO] GPU available: {torch.cuda.get_device_name(0)} (Count: {torch.cuda.device_count()})")
    else:
        print("[INFO] No GPU detected. Ollama might use CPU or internal GPU if supported.")

    print(f"[INFO] Pulling Ollama model '{args.model}'...")
    try:
        subprocess.run(["ollama", "pull", args.model], check=True, capture_output=True, text=True)
        print("[INFO] Model pulled or already available.")
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed to pull model '{args.model}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    image_dir = args.image_dir if os.path.isabs(args.image_dir) else os.path.join(os.getcwd(), args.image_dir)
    print(f"[INFO] Searching images in: {image_dir}")

    images = get_image_paths(image_dir)
    if not images:
        print(f"[ERROR] No images found in {image_dir}. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(images)} images.")

    #Output file
    with open(args.output_file, 'w') as f:
        f.write(f"Image Captioning Results\nModel: {args.model}\nPrompt: {args.prompt}\n{'='*60}\n")

    print("[INFO] Starting captioning...")
    for img_path in tqdm(images, desc="Captioning Images", unit="image"):
        caption_image(img_path, args.model, args.prompt, args.output_file)

    print(f"[INFO] Captioning complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
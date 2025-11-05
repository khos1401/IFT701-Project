import os
from pathlib import Path
from image_augmentation import JpgToPngProcessor, ImageOpsConfig

def process_folder(input_dir: str, output_dir: str, config: ImageOpsConfig):
    """
    Process all JPG/JPEG images in a folder and save them as reduced PNGs
    to a new folder.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.jpg"):
        _process_file(file, output_path, config)
    for file in input_path.glob("*.jpeg"):
        _process_file(file, output_path, config)


def _process_file(file_path: Path, output_path: Path, config: ImageOpsConfig):
    output_file = output_path / f"{file_path.stem}.png"
    try:
        processor = JpgToPngProcessor(str(file_path))
        processor.process(config, output_path=str(output_file), return_bytes=False)
        print(f"✅ Processed: {file_path.name} → {output_file.name}")
    except Exception as e:
        print(f"⚠️ Skipped {file_path.name}: {e}")


if __name__ == "__main__":
    # Adjust your parameters here
    config = ImageOpsConfig(
        max_size=(16, 16),           # fit within 16x16 while keeping aspect ratio
        center_crop_size=(16, 16)    # optional center crop (remove if not needed)
    )

    # Input folders
    input_folders = ["dogs", "cats"]
    output_folders = ["reduced_dogs_16", "reduced_cats_16"]
    
    for inp, out in zip(input_folders, output_folders):
        if not os.path.exists(inp):
            print(f"⚠️ Folder '{inp}' does not exist, skipping.")
            continue
        print(f"Processing folder: {inp}")
        process_folder(inp, out, config)
        print(f"✅ Finished: {inp} → {out}")

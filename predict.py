
import argparse
import urllib.request
from pathlib import Path

import cv2


# Project folders
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "results"

MODEL_PATH = MODEL_DIR / "ESPCN_x4.pb"

# Pre-trained ESPCN model
MODEL_URL = (
    "https://github.com/fannymonori/TF-ESPCN/"
    "raw/master/export/ESPCN_x4.pb"
)


def download_model():
    """Download the pre-trained model if it is missing."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print("Downloading pre-trained ESPCN model...")

        try:
            urllib.request.urlretrieve(
                MODEL_URL,
                str(MODEL_PATH)
            )
        except Exception as error:
            raise RuntimeError(
                "Model download failed. Check your internet connection."
            ) from error

    if MODEL_PATH.stat().st_size == 0:
        raise RuntimeError("Downloaded model file is empty.")

    print("Model file is ready.")


def enhance_image(input_path, output_path):
    """Upscale an image by 4 times using ESPCN."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {input_path}"
        )

    # Load input image
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            "Cannot read image. Check the image format."
        )

    original_height, original_width = image.shape[:2]

    # Load ESPCN model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(MODEL_PATH))
    sr.setModel("espcn", 4)

    # Enhance image
    enhanced_image = sr.upsample(image)

    # Create output folder
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save enhanced image
    success = cv2.imwrite(
        str(output_path),
        enhanced_image
    )

    if not success:
        raise RuntimeError("Could not save output image.")

    enhanced_height, enhanced_width = enhanced_image.shape[:2]

    print("\nImage enhancement completed!")
    print(f"Original size: {original_width} x {original_height}")
    print(f"Enhanced size: {enhanced_width} x {enhanced_height}")
    print(f"Saved output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Image Super-Resolution using Deep Learning"
    )

    parser.add_argument(
        "--input",
        default=str(INPUT_DIR / "test.png"),
        help="Path of input image"
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "super_resolution_output.png"),
        help="Path of output image"
    )

    args = parser.parse_args()

    try:
        download_model()
        enhance_image(args.input, args.output)

    except Exception as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

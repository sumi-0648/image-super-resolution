
from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
import cv2
import urllib.request
import numpy as np
import threading
import uuid

app = Flask(__name__)

# Limit uploaded image size to 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Reduce OpenCV memory and CPU usage
cv2.setNumThreads(1)
cv2.setUseOptimized(True)

# Project folders
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "input_images"
RESULT_DIR = BASE_DIR / "results"

MODEL_PATH = MODEL_DIR / "ESPCN_x4.pb"

MODEL_URL = (
    "https://github.com/fannymonori/TF-ESPCN/"
    "raw/master/export/ESPCN_x4.pb"
)

for folder in [MODEL_DIR, UPLOAD_DIR, RESULT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Maximum input image dimension
MAX_IMAGE_SIDE = 512

# Load the model only once
sr_model = None
model_lock = threading.Lock()


def load_model():
    global sr_model

    with model_lock:
        if sr_model is not None:
            return sr_model

        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            print("Downloading ESPCN model...")

            with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                MODEL_PATH.write_bytes(response.read())

        if MODEL_PATH.stat().st_size == 0:
            raise RuntimeError("AI model file is empty.")

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(MODEL_PATH))
        sr.setModel("espcn", 4)

        sr_model = sr
        print("ESPCN model loaded successfully.")

        return sr_model


def enhance_image(image, sr):
    # AI-based 4x super resolution
    output = sr.upsample(image)

    # Improve contrast
    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    output = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    # Slight brightness improvement
    output = cv2.convertScaleAbs(
        output,
        alpha=1.05,
        beta=5
    )

    # Sharpen image
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    output = cv2.filter2D(output, -1, kernel)

    return output


@app.route("/", methods=["GET", "POST"])
def home():
    original = None
    enhanced = None
    original_size = None
    enhanced_size = None
    error = None

    if request.method == "POST":
        try:
            file = request.files.get("image")

            if not file or not file.filename:
                raise ValueError("Please select an image.")

            filename = secure_filename(file.filename)

            if Path(filename).suffix.lower() not in [
                ".png", ".jpg", ".jpeg"
            ]:
                raise ValueError("Please upload PNG or JPG image.")

            # Use unique filenames to avoid overwriting uploads
            unique_id = uuid.uuid4().hex
            input_name = f"{unique_id}_{filename}"
            output_name = f"{unique_id}_enhanced.png"

            input_path = UPLOAD_DIR / input_name
            output_path = RESULT_DIR / output_name

            file.save(str(input_path))

            image = cv2.imread(str(input_path))

            if image is None:
                raise ValueError("Unable to read the uploaded image.")

            original_size = f"{image.shape[1]} x {image.shape[0]}"

            # Resize large images before processing
            height, width = image.shape[:2]
            largest_side = max(height, width)

            if largest_side > MAX_IMAGE_SIDE:
                scale = MAX_IMAGE_SIDE / largest_side

                new_width = max(1, int(width * scale))
                new_height = max(1, int(height * scale))

                image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                )

            # Load cached AI model
            sr = load_model()

            # Enhance image (4x only)
            output = enhance_image(image, sr)

            success = cv2.imwrite(str(output_path), output)

            if not success:
                raise RuntimeError("Unable to save enhanced image.")

            enhanced_size = f"{output.shape[1]} x {output.shape[0]}"

            original = f"/input/{input_name}"
            enhanced = f"/result/{output_name}"

            # Release temporary image arrays
            del image
            del output

        except Exception as exc:
            error = str(exc)
            print("Image enhancement error:", error)

    return render_template(
        "index.html",
        original=original,
        enhanced=enhanced,
        original_size=original_size,
        enhanced_size=enhanced_size,
        error=error
    )


@app.route("/input/<filename>")
def input_image(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/result/<filename>")
def result_image(filename):
    return send_from_directory(str(RESULT_DIR), filename)


@app.errorhandler(413)
def file_too_large(error):
    return "Image is too large. Please upload an image under 5 MB.", 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

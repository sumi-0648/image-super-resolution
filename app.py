
from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
import cv2
import urllib.request
import numpy as np

app = Flask(__name__)

# Project folders
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "input_images"
RESULT_DIR = BASE_DIR / "results"

# ESPCN pre-trained model
MODEL_PATH = MODEL_DIR / "ESPCN_x4.pb"

MODEL_URL = (
    "https://github.com/fannymonori/TF-ESPCN/"
    "raw/master/export/ESPCN_x4.pb"
)

# Create folders automatically
for folder in [MODEL_DIR, UPLOAD_DIR, RESULT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# Load AI model
def load_model():

    if not MODEL_PATH.exists():

        print("Downloading AI Super Resolution Model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            str(MODEL_PATH)
        )

    if MODEL_PATH.stat().st_size == 0:
        raise RuntimeError("AI model file is empty.")

    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    sr.readModel(str(MODEL_PATH))

    sr.setModel("espcn", 4)

    return sr


# Image enhancement function
def enhance_image(image, sr):

    # Step 1: AI-based 4x super resolution
    output = sr.upsample(image)

    # Step 2: Additional 2x enlargement
    output = cv2.resize(
        output,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Step 3: Brightness and contrast enhancement
    lab = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2LAB
    )

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

    # Step 4: Improve brightness slightly
    output = cv2.convertScaleAbs(
        output,
        alpha=1.10,
        beta=10
    )

    # Step 5: Improve sharpness
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    output = cv2.filter2D(
        output,
        -1,
        kernel
    )

    return output


# Main webpage
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

            # Check uploaded file
            if not file or not file.filename:
                raise ValueError(
                    "Please select an image."
                )

            filename = secure_filename(file.filename)

            # Check image extension
            if Path(filename).suffix.lower() not in [
                ".png", ".jpg", ".jpeg"
            ]:
                raise ValueError(
                    "Please upload PNG or JPG image."
                )

            # Save uploaded image
            input_path = UPLOAD_DIR / filename

            output_path = RESULT_DIR / "enhanced.png"

            file.save(str(input_path))

            # Read original image
            image = cv2.imread(str(input_path))

            if image is None:
                raise ValueError(
                    "Unable to read the uploaded image."
                )

            # Original image dimensions
            original_size = (
                f"{image.shape[1]} x {image.shape[0]}"
            )

            # Load AI model
            sr = load_model()

            # Enhance image
            output = enhance_image(image, sr)

            # Save enhanced image
            success = cv2.imwrite(
                str(output_path),
                output
            )

            if not success:
                raise RuntimeError(
                    "Unable to save enhanced image."
                )

            # Enhanced image dimensions
            enhanced_size = (
                f"{output.shape[1]} x {output.shape[0]}"
            )

            original = f"/input/{filename}"

            enhanced = "/result/enhanced.png"

        except Exception as exc:

            error = str(exc)

    return render_template(
        "index.html",
        original=original,
        enhanced=enhanced,
        original_size=original_size,
        enhanced_size=enhanced_size,
        error=error
    )


# Display original image
@app.route("/input/<filename>")
def input_image(filename):

    return send_from_directory(
        str(UPLOAD_DIR),
        filename
    )


# Display enhanced image
@app.route("/result/<filename>")
def result_image(filename):

    return send_from_directory(
        str(RESULT_DIR),
        filename
    )


# Run Flask application
if __name__ == "__main__":

    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
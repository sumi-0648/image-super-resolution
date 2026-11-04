# 🖼️ Image Super Resolution Using Deep Learning

## 📌 Project Overview
Image Super Resolution is a Deep Learning-based project that enhances low-resolution images into high-resolution images using the ESPCN (Efficient Sub-Pixel Convolutional Neural Network) model.

The project improves image resolution by scaling images up to 4× their original dimensions.

## 🎯 Objectives
- Enhance low-resolution images.
- Generate high-resolution images.
- Improve image clarity and quality.
- Demonstrate Deep Learning applications in image processing.

## 🛠️ Technologies Used
- Python
- OpenCV
- NumPy
- Pillow
- Matplotlib
- ESPCN Deep Learning Model
- Flask (if using the web application)

## 📂 Project Structure

```text
image-super-resolution/
│
├── models/
│   └── ESPCN_x4.pb
├── input_images/
├── results/
├── predict.py
├── Image_Super_Resolution.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sumi-0648/image-super-resolution.git
```

Move into the project directory:

```bash
cd image-super-resolution
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

Run the prediction script:

```bash
python predict.py --input input_images/test.png
```

The enhanced image will be saved in the results folder.

## ✨ Features
- 4× Image Upscaling
- Deep Learning-based Image Enhancement
- Easy Image Processing
- High-Resolution Image Generation

## 🚀 Future Improvements
- Implement advanced super-resolution architectures.
- Improve image quality and performance.
- Develop a user-friendly web interface.

## 👩‍💻 Author
**Sumi Sharon**

GitHub: https://github.com/sumi-0648

## 📜 License
This project is available for educational and learning purposes.

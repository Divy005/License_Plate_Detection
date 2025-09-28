# 🚗 License Plate Detection with YOLOv8

An AI-powered license plate detection system built using YOLOv8 and deployed with Streamlit. This project can accurately detect and locate license plates in images using deep learning.

## 📋 Project Description

This project implements an end-to-end license plate detection system that:
- Uses YOLOv8n (nano) architecture for fast and efficient detection
- Trained on a custom dataset with 7,057 training images and 2,047 validation images
- Provides a user-friendly Streamlit web interface for easy interaction
- Supports common image formats (JPG, JPEG, PNG)
- Visualizes detection results with bounding boxes

## 🏗️ Architecture

The system consists of two main components:
1. **Model Training Pipeline**: Custom YOLOv8 model trained specifically for license plate detection
2. **Web Application**: Streamlit-based interface for real-time detection

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/license-plate-detection.git
cd license-plate-detection

# Install dependencies
pip install ultralytics streamlit opencv-python pillow numpy

# For training (if using Google Colab)
pip install ultralytics
```

## 📊 Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## 🚀 Usage

### Running the Web Application
```bash
streamlit run app.py
```

### Training the Model (Google Colab)
1. Upload your dataset to Google Drive
2. Run the provided Jupyter notebook (`license_plate_detection.ipynb`)
3. The trained model will be saved as `best.pt`

### Model Training Configuration
- **Architecture**: YOLOv8n (3M parameters)
- **Input Size**: 640x640 pixels
- **Epochs**: 50
- **Batch Size**: 16
- **Optimizer**: AdamW with automatic parameter optimization

## 📁 File Structure

```
license-plate-detection/
├── app.py                          # Streamlit web application
├── license_plate_detection(2).ipynb   # Training notebook
├── best.pt                         # Trained model weights
├── dataset.zip                     # Training dataset (if included)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🎯 Features

- **Real-time Detection**: Fast inference using YOLOv8n architecture
- **High Accuracy**: Trained on diverse license plate dataset
- **User-Friendly Interface**: Simple drag-and-drop web interface
- **Visualization**: Clear bounding box visualization of detected plates
  

## 🔧 Configuration

### Data Configuration (data.yaml)
```yaml
train: /path/to/dataset/train/images
val: /path/to/dataset/valid/images
test: /path/to/dataset/test/images

nc: 1
names: ['license_plate']
```

### Model Path Configuration
Update the model path in `app.py`:
```python
model_path = "path/to/your/best.pt"
```

## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment
- **Streamlit Cloud**: Connect your GitHub repo for automatic deployment
- **Heroku**: Use the provided `Procfile` for Heroku deployment
- **Docker**: Containerize the application for consistent deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request
   
## 📸 Image
<img width="2416" height="1280" alt="image" src="https://github.com/user-attachments/assets/bfdd81d0-0c4a-4311-8192-cdc6a325721a" />

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base architecture
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenCV](https://opencv.org/) for image processing capabilities

## 📧 Contact

For questions or suggestions, please open an issue or contact [divydobariya11@gmail.com]

---

⭐ If you found this project helpful, please consider giving it a star!

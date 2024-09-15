# Binary-Multi-CNN-CIFAR10
**CNN Model for Binary & Multi-Class Classification using CIFAR-10 Dataset**

## Overview
This project showcases an image classification application built using TensorFlow and convolutional neural networks (CNNs) to classify images from the CIFAR-10 dataset. It includes two models:
- **Binary Classification Model**: Classifies images into two categories.
- **Multi-Class Classification Model**: Classifies images into ten different classes (e.g., dog, cat, horse).

## Key Features
- Utilizes the CIFAR-10 dataset for training and testing.
- Implements two CNN models for binary and multi-class classification.
- Integrated into a Flask web application for easy image upload and classification.
- Potential for further enhancements like real-time classification and improved user interface.

## Project Structure
```
Binary-Multi-CNN-CIFAR10/
│
├── app.py                      # Flask application for model deployment
├── model_binary.h5             # Pre-trained binary classification model
├── model_multiclass.h5         # Pre-trained multi-class classification model
├── static/                     # Static files (e.g., CSS, images)
│   └── styles.css              # Stylesheet for the web application
├── templates/                  # HTML templates for Flask
│   ├── index.html              # Homepage for image upload
│   └── result.html             # Result page for displaying predictions
├── README.md                   # Project documentation (this file)
└── requirements.txt            # Required packages
```

## Getting Started

### Prerequisites
Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### Install Required Packages
Install Flask and TensorFlow using the following command:
```bash
pip install Flask tensorflow
```

### Organize Files
Place the provided code in the appropriate files and directories as described in the project structure.

### Load Models
Ensure the paths to your trained models are correctly specified in `app.py`:
```python
model_binary = load_model('model_binary.h5')
model_multiclass = load_model('model_multiclass.h5')
```

## Running the Application
1. **Navigate to the project directory**:
    ```bash
    cd Binary-Multi-CNN-CIFAR10
    ```
2. **Run the Flask application**:
    ```bash
    python app.py
    ```
3. Open your browser and go to `http://127.0.0.1:5000/` to access the application.

## Models Overview

### Data Loading and Preprocessing
- The CIFAR-10 dataset is loaded and normalized to a range between 0 and 1 for efficient model training.

### Multi-Class Classification Model
- A CNN model is constructed to classify images into ten different classes.

### Binary Classification Model
- A separate CNN model is built to classify images into two categories, demonstrating the adaptability of CNNs for binary tasks.

## Future Enhancements
This project serves as a foundation for further development and enhancements:

1. **Flask Web Application**: Enhance the Flask app to allow users to upload images and receive classification predictions.
2. **Improved User Interface**: Utilize Bootstrap and CSS to create an engaging frontend.
3. **Additional Classes**: Expand the model to recognize a wider variety of objects.
4. **Real-Time Classification**: Implement real-time image classification using a device's camera.
5. **Deployment to Production**: Deploy the web application to a public server.
6. **User Authentication**: Add user-specific dashboards for managing classification history.
7. **Model Fine-Tuning**: Experiment with different architectures and hyperparameters to improve accuracy.

## Usage
Upload an image through the web interface to get predictions from the binary and multi-class models.

## Screenshots
**Home Page:**
![Home Page](assets/home_page_screenshot.png)

**Prediction Results:**
![Prediction Results](assets/result_screenshot.png)

## Credits
This project is created and maintained by **[Abhishek Shah]**.

## License
This project is licensed under the [MIT License](LICENSE).

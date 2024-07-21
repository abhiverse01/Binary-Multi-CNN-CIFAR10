# Binary-Multi-CNN-CIFAR10
**CNN-Model-for-Binary & Multi-Class-Classification-using-CIFAR-10-Dataset**
- Contains my project code for two CNN models, one trained for binary classification while the other made for multi-class classification. It utillises the CIFAR-10 dataset. 

- This project showcases an image classification application built using TensorFlow. The application utilizes convolutional neural networks (CNNs) to classify images from the CIFAR-10 dataset into multiple classes for multi-class classification and into two classes for binary classification.

# Project Overview
- The project involves the following main components:

## Data Loading and Preprocessing: 
- The CIFAR-10 dataset is loaded, and the images are normalized to a range between 0 and 1.

## Multi-Class Classification Model: 
- A CNN model is constructed for multi-class classification, where images are classified into ten different classes (e.g., dog, cat, horse).

## Binary Classification Model: 
- Another CNN model is built for binary classification, classifying images into two categories.

## Install Required Packages:
```bash
pip install Flask tensorflow
```

## Organize Files:
- Place the provided code in appropriate files and directories as described in the project structure.

## Load Models:
- Make sure to replace 'model_path.h5' with the actual paths to your trained models.

## Run the Application:
- Navigate to the project directory and run the app:
```bash
cd theimagedetectorproject
```
- Run it through the terminal:
```bash
python app.py
```

# Future Enhancements
- This project serves as a foundation for further development and enhancements:

1. Flask Web Application: The models are integrated into a Flask web application that allows users to upload images and receive classification predictions.
2. Improved User Interface: Enhance the frontend design using Bootstrap and CSS to create an engaging and visually appealing user interface.
3. Additional Classes: Expand the classification models to include more classes, enabling the application to recognize a wider variety of objects.
4. Real-Time Classification: Implement real-time image classification using a device's camera, enabling users to take pictures and receive instant predictions.
5. Deployment to Production: Deploy the web application to a production server for public access, and ensure scalability and performance optimizations.
6. User Authentication: Implement user authentication and user-specific dashboards to save and manage image classification history.
7. Model Fine-Tuning: Experiment with different model architectures, hyperparameters, and training strategies to improve classification accuracy.

- Feel free to explore, customize, and expand upon this project to create a feature-rich and versatile image classification web application.

# Credits
- This project is created and maintained by **[Abhishek Shah]**

# License
- This project is licensed under the [MIT License]

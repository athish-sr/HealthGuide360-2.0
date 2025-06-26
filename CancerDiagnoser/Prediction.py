import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2,json

# Load models
lung_cancer_model = tf.keras.models.load_model('CancerDiagnoser\\lung_cancer_model.h5')
skin_cancer_model = tf.keras.models.load_model('CancerDiagnoser\\skin_cancer_model.h5')
blood_cancer_model = tf.keras.models.load_model('CancerDiagnoser\\blood_cancer_model.h5')
kidney_cancer_model = tf.keras.models.load_model('CancerDiagnoser\\kidney_cancer_model.h5')
brain_tumor_model = tf.keras.models.load_model('CancerDiagnoser\\brain_tumor_model.h5')

# Class labels for each model
lung_classes = ['Normal', 'Benign', 'Malignant']
skin_classes = ['Benign', 'Malignant']
blood_classes = ['[Malignant] early Pre-B', '[Malignant] Pre-B', '[Malignant] Pro-B', 'Benign']
kidney_classes = ['Cyst', 'Normal', 'Stone', 'Tumor']
brain_classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

with open("CancerDiagnoser\\response.json",'r') as f:
    details=json.load(f)    
f.close()
# Function to preprocess images
def load_and_preprocess_image(img_path, target_size, is_gray=False):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The image at path {img_path} does not exist.")
    
    if is_gray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is None:
            raise ValueError(f"Failed to load image at path {img_path}.")
        img = cv2.resize(img, target_size)
        img_array = img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    else:
        img = image.load_img(img_path, target_size=target_size)  # Load as color
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction functions for each cancer type with confidence threshold
def predict_lung_cancer(image_path):
    img_array = load_and_preprocess_image(image_path, (128, 128), is_gray=True)
    prediction = lung_cancer_model.predict(np.expand_dims(img_array, axis=0))
    max_confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)

    # Ensure you're using the correct class labels for lung cancer
    predicted_class = lung_classes[predicted_class_index]
    # Check confidence threshold
    if max_confidence > 0.80:
        detail = details["Lung Cancer"][predicted_class]
    else:
        detail = {"description": None , "guidance": "Consider consulting a healthcare professional for further evaluation."}
    print(detail)  
    return predicted_class, max_confidence,detail


def predict_skin_cancer(image_path):
    img_array = load_and_preprocess_image(image_path, (150, 150))
    prediction = skin_cancer_model.predict(img_array)
    max_confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)

    # Ensure you're using the correct class labels for lung cancer
    predicted_class = skin_classes[predicted_class_index]
    # Check confidence threshold
    if max_confidence > 0.70:
        detail = details["Skin Cancer"][predicted_class]
    elif max_confidence<0.30:
        detail=details["Skin Cancer"]["Benign"]
    else:
        detail = {"description": None , "guidance": "Consider consulting a healthcare professional for further evaluation."}
    print(detail)  
    return predicted_class, max_confidence,detail

def predict_blood_cancer(image_path):
    img_array = load_and_preprocess_image(image_path, (150, 150))
    prediction = blood_cancer_model.predict(img_array)
    max_confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)

    # Ensure you're using the correct class labels for lung cancer
    predicted_class = blood_classes[predicted_class_index]
    # Check confidence threshold
    if max_confidence > 0.70:
        detail = details["Blood Cancer"][predicted_class]
    else:
        detail = {"description": None , "guidance": "Consider consulting a healthcare professional for further evaluation."}
    print(detail)  
    return predicted_class, max_confidence,detail

def predict_kidney_cancer(image_path):
    img_array = load_and_preprocess_image(image_path, (128, 128), is_gray=True)
    prediction = kidney_cancer_model.predict(np.expand_dims(img_array, axis=0))
    max_confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)

    # Ensure you're using the correct class labels for lung cancer
    predicted_class = kidney_classes[predicted_class_index]
    # Check confidence threshold
    if max_confidence > 0.80:
        detail = details["Kidney Cancer"][predicted_class]
    else:
        detail = {"description": None , "guidance": "Consider consulting a healthcare professional for further evaluation."}
    print(detail)  
    return predicted_class, max_confidence,detail

def predict_brain_cancer(image_path):
    img_array = load_and_preprocess_image(image_path, (128, 128))
    prediction = brain_tumor_model.predict(img_array)
    max_confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)

    # Ensure you're using the correct class labels for lung cancer
    predicted_class = brain_classes[predicted_class_index]
    # Check confidence threshold
    if max_confidence > 0.75:
        detail = details["Brain Cancer"][predicted_class]
    else:
        detail = {"description": None , "guidance": "Consider consulting a healthcare professional for further evaluation."}
    print(detail)  
    return predicted_class, max_confidence,detail

# Image paths for testing
# lung_image_path = r"D:\providence\archive (7)\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\Malignant\Malignant case (20).jpg"
# skin_image_path = r"D:\providence\archive (9)\melanoma_cancer_dataset\test\benign\melanoma_9610.jpg"
# blood_image_path = r"D:\providence\archive (10)\Blood cell Cancer [ALL]\[Malignant] Pro-B\Snap_003.jpg"
# kidney_image_path = r"C:\Users\valan\OneDrive\Pictures\Screenshots\Screenshot (5).png"
# brain_image_path = r"C:\Users\valan\OneDrive\Pictures\Screenshots\Screenshot (5).png"

# # Print predictions
# try:
#     print(predict_lung_cancer(lung_image_path))
# except Exception as e:
#     print(e)

# try:
#     print(predict_skin_cancer(skin_image_path))
# except Exception as e:
#     print(e)

# try:
#     print(predict_blood_cancer(blood_image_path))
# except Exception as e:
#     print(e)

# try:
#     print(predict_kidney_cancer(kidney_image_path))
# except Exception as e:
#     print(e)

# try:
#     print(predict_brain_tumor(brain_image_path))
# except Exception as e:
#     print(e)

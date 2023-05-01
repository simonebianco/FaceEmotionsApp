import streamlit as st
import tensorflow as tf
import time
from PIL import Image
import numpy as np
import cv2

# FUNCTIONS
def load_model(path_model, path_weight):
    json_file = open(path_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(path_weight)
    return model  
    
def predict_emotion(image, model, emotions):
        predictions = model.predict(image)
        emotion_pred = emotions[np.argmax(predictions)]
        return emotion_pred

def detect_faces(our_image, model, emotions, face_cascade):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	for (x, y, w, h) in faces:
			fc = gray[y:y+h, x:x+w]
			roi = cv2.resize(fc, (48, 48))            
			pred = predict_emotion(roi[np.newaxis, :, :, np.newaxis], model, emotions)
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
			return img, faces, pred 


# Load Model
path_model = 'model.json'
path_weight = 'weight.h5'
emotions = ['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']

model = load_model(path_model, path_weight)

st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX 

# titolo pagina
st.title('Face Emotions App')
st.markdown("____")

# Image Uploader
image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

# Original Image
if image_file is not None:
    our_image = Image.open(image_file)
    st.markdown("**Original Image**")
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i+1)
    st.image(our_image)

if image_file is None:
    st.error("No image uploaded yet")

# Process Button
if st.button("Analyze"):
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i+1)

    result_img ,result_faces ,prediction = detect_faces(our_image, model, emotions, face_cascade)    

    if st.image(result_img):
        st.success("Found {} faces".format(len(result_faces)))
        pred_final = "Emotion predicted: {}!".format(prediction)
        st.subheader(pred_final)
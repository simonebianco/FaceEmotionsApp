import streamlit as st
import pandas as pd

# Classification Report
class_report = pd.read_csv('utils\classification_report.csv')
class_report.rename(columns={"Unnamed: 0": "features"}, inplace=True)
# History
train_history = pd.read_csv('utils\history.csv')
train_history.rename(columns={"Unnamed: 0": "Epochs"}, inplace=True)


# Titolo Pagina
st.title('Face Emotions App')
st.markdown("____")
st.subheader("Model Performance")

# Selectbox
actions = ["Model Architecture",
           "Train and Test Accuracy",          
           "Classification Report" , 
           "Confusion Matrix",
           "Loss and Learning Rate",
           "Training History"
           ]

action = st.selectbox("Choose One", actions)

if action == 'Model Architecture':
    st.subheader('**Model Architecture**')
    st.image('utils\model_architecture.png', width=700)

if action == 'Train and Test Accuracy':
    st.subheader('**Train and Test Accuracy**')
    st.image('utils\loss_accuracy.png', width=800)

if action == 'Classification Report':
    st.subheader('**Classification Report**')
    st.write(class_report)  

if action == 'Confusion Matrix':
    st.subheader('**Confusion Matrix**')
    st.image('utils\confusion_matrix.png', width=800)

if action == 'Loss and Learning Rate':
    st.subheader('**Loss and Learning Rate**')
    st.image('utils\lrs.png', width=700)  

if action == 'Training History':
    st.subheader('**Training History**')
    st.write(train_history)               
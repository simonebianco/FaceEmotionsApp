
import streamlit as st

# Titolo Pagina
st.title('Face Emotions App')
st.markdown("____")
st.subheader("Home")


text = 'The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.'
st.write(text)

st.image('utils\images_examples.png', width=700)


# Link Kaggle Data
st.write('Link dataset at: https://www.kaggle.com/datasets/msambare/fer2013')


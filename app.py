import streamlit as st
import pandas as pd
import torch
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.markdown("<h1 style='text-align: center;'>Gender Classification</h1>", unsafe_allow_html=True)

# input image
slf = st.camera_input(label="")
file = slf
if slf == None:
    st.markdown("<h5 style='text-align: center;'>or</h1>", unsafe_allow_html=True)
    upl = st.file_uploader(label="", type=['png', 'jpeg', 'jpg'])
    file = upl

if file:
    # PIL convert
    img = PILImage.create(file)

    # model
    model1 = load_learner('humans_vs_non_humans_model.pkl')
 
    # prediction
    pred1, pred_id1, prob1 = model1.predict(img)

    # check for human
    if pred1 == "Human":
        model2 = load_learner('humans.pkl')
        pred2, pred_id2, prob2 = model2.predict(img)

        # create dataframe
        data={'Gender': ['Female', 'Male'], 'Accuracy by percentage': [prob2.tolist()[0]*100, prob2.tolist()[1]*100]}
        df = pd.DataFrame(data=data)

        # plot
        fig = px.bar(
            df, 
            x='Accuracy by percentage',
            y='Gender',
            height=300,
            text_auto='.2s'
        )
        st.image(file, width=768)      
        st.plotly_chart(fig)
    else:
        st.image(file, width=768)      
        st.error("The image does not include a human!")

# example images
st.markdown("<h3 style='text-align: center; color:green;'>Example photos</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.image('./examples/male.jpg', caption="Male")
col2.image('./examples/female.jpeg', caption="Female")

# more info
options = [
    'Select',
    'What Data Set did I use?',
    'How does it differentiate between humans and non-humans?'
]
option = st.selectbox('How I made it?', options, index=0)

if option == 'What Data Set did I use?':
    body = """
    - [Male and female faces dataset](https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset) provided by > Kaggle
    - [Open Image Dataset v4](https://storage.googleapis.com/openimages/web/index.html) which provides [600](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html) classes
    """
    st.markdown(body)

if option == 'How does it differentiate between humans and non-humans?':
    body = """
    I built 2 models. The first model differtiates humans from non-humans. If the given image includes a human, then the second model classifies the image by gender:
    1. **[Human recognition model](https://github.com/Jakhongir0103/Gender_classification_model/blob/main/Human_classification_model.ipynb)** trained using 2 datasets:
        - 500 images from the class Animal of the [Dataset](https://storage.googleapis.com/openimages/web/index.html)
        - 500 images from the Male/Female [Dataset](https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset) 
    2. **[Gender Classification model](https://github.com/Jakhongir0103/Gender_classification_model/blob/main/Gender_classification_model.ipynb)** trained using 2 datasets:
        - ~1500 images of Males
        - ~1500 images of Females
    """
    st.markdown(body)

if option == 'What Data Base did I use?':
    st.markdown(body)

if option == 'What Data Base did I use?':
    st.write(option)

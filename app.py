import numpy as np
import pickle
import pandas as pd
import streamlit as st
import sklearn.linear_model
from PIL import Image



pickle_in = open("linear.pkl", "rb")
model = pickle.load(pickle_in)



def welcome():
    return "Welcome All"



def predict_life_satisfaction(GDP_country):
    """Let's predict life satisfaction
    This is using docstrings for specifications.
    ---
    parameters:
      - name: GDP_country
        in: query
        type: number
        required: true

    responses:
        200:
            description: The output values

    """

    prediction = model.predict(np.array([[GDP_country]], dtype='float64'))
    print(prediction.flatten())
    return prediction.flatten()


def main():
    st.title("Life Satisfaction Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Life Satisfaction Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    GDP_country = st.text_input("GDP of Country", "Type Here GDP Vlaue, for eg: 22,587")
    result = ""
    if st.button("Predict"):
        result = predict_life_satisfaction(GDP_country)
    st.success('Life Satisfaction is : {}'.format(result))
    if st.button("About"):
        st.text("In this app we are predicting Life Satisfaction using GDP of a particular Country")
        st.text("Equation representing model:")
        st.latex('lifesatisfaction = θ_0+ θ_1× GDP per capita')
        st.latex('where : θ_0 = 4.85 , θ_1 = 4.91 × 10^-5')
        st.text("Built with Streamlit By Nitin Rajput")

    """
    ## Table 1-1. Does money make people happier? 
    """

    df = pd.DataFrame({
        'GDP per capita(USD)': ['12,240','27,195','37,675','50,962','55,805'],
        'Life Satisfaction': [4.9, 5.8, 6.5, 7.4,7.2],'Country':['Hungary','Korea','France','Australia','United States']})

    df.set_index('Country',inplace=True)

    df

    image = Image.open('plot.jpg')
    st.image(image, caption='Scatter Plot with fitted line',use_column_width = True)






if __name__ == '__main__':
    main()
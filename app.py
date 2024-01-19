import pandas as pd 
import numpy as np 
import streamlit as st
import joblib

Model = joblib.load('Model.h5')

st.title('Customer Segmentation ')
st.info('(Recency): Number of days since last purchase')
st.info('(Frequency): Number of tracsactions')
st.info('(Monetary): Total amount of transactions ')
st.image('rfmgf.png')
          


def Predict(Recency,Frequancy,Monetary):
    
    
    test = pd.DataFrame(data=[ [Recency,Frequancy,Monetary] ] , 
                       columns=['Recency','Frequancy','Monetary'])
    
    return Model.predict(test)


def main():
    Recency= st.slider(f'Select Recency ', min_value = 0.0, max_value = 1165.0, step = 1.0 )
    
    Frequancy= st.slider(f'Select Frequancy ', min_value = 1.0, max_value = 35.0, step = 1.0)
    
    Monetary= st.slider(f'Select Monetary ', min_value = 4.0, max_value = 25045.0, step = 10.0 )
    
    if st.button('Predict'):
        
        reply = Predict(Recency,Frequancy,Monetary)
        st.write(reply)
        
main()     


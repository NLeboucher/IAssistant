import streamlit as st

import pandas as pd

# importing the requests library
import requests
  
# api-endpoint
URL = "http://localhost/"
  
# location given here
location = "delhi technological university"
  
# defining a params dict for the parameters to be sent to the API
PARAMS = {'address':location}
  
# sending get request and saving the response as response object
r = requests.get(url = URL, params = PARAMS)

st.title("Model Tests:")
st.title("  MPNet")
aa=st.text_input("Premise for MPNet.")
ab=st.text_input("Hypothesis for MPNet.")
st.write("Predicted link : Entailment")


st.title("  LSTM")
ba=st.text_input("Premise for LSTM")
bb=st.text_input("Hypothesis for LSTM")
st.write("Predicted Link : Contradiction")

st.title("Parameter change Test:")
temp=st.text_input("Premise for MPNet")
st.write("MPNet : Predicted parameter change: sun 30")
st.write("LSTM : Predicted parameter change: sun 30")
import streamlit as st
import pandas as pd
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 45]
})
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
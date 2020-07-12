import streamlit as st

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import pickle

# load model
model = pickle.load(open('model.pkl','rb'))

# apresentar números com 3 casas decimais
pd.set_option('display.float_format', lambda x: '%.0f' % x)


# Text/Title
st.title("Probabilidade de furto do seu veículo")

hora_leit = st.number_input(label="Hora da leitura")
periodo_leit = st.number_input(label="Periodo da leitura")
hora_gprs = st.number_input(label="Hora da gprs")
periodo_gprs = st.number_input(label="Periodo da gprs")
peso = st.number_input(label="Peso")


all_models = ['HB20','ONIX','GOL','Saveiro']
model_choice = st.selectbox("Modelo",all_models)

ano = st.text_input("Ano de fabricação do veículo")

df_prob = pd.DataFrame({'hora_leit': hora_leit,
                        'periodo_leit': periodo_leit,
                        'hora_gprs': hora_gprs,
                        'periodo_gprs': periodo_gprs, 
                        'peso': peso,
                        'temp' : [0] })

del df_prob['temp']

df_prob.head()

if st.button("Submeter"):
    probability_class_1 = model.predict_proba(df_prob)[0, 1]
    st.write(probability_class_1*100)  
	

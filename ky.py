import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings
#warnings.filterwarnings('ignore')
# 1.基础配置
# 加载训练好的随机森林模型(确保rf.pkl与脚本同目录)
model = joblib.load('./model/LR2.pkl')
# 加载测试数据
X_test = pd.read_csv('./model/test_set.csv')
# 定义特征名称(替换为业务相关列名，与编码规则对应)
feature_names = model.feature_names_in_

# Streamlit页面配置和特征输入组件
import streamlit as st
warnings.filterwarnings('ignore', message='Unable to initialize signal')

# 在Streamlit应用中使用
st.set_page_config(page_title="Mechanical Ventilation Risk Predictor", layout="wide")
st.title("Mechanical Ventilation Risk Predictor for Patients with Acute Pulmonary Embolism")
st.markdown("#### Please fill in the information below and click to get MV prediction results")

# Dropdown input
oxygen_saturation = st.selectbox("oxygen_saturation＜90%", ("Yes", "No"))

# Input bar 1
CT_PMA_10th = st.number_input("Enter CT_PMA_10th(HU)")

# Dropdown input
smoke = st.selectbox("smoke", ("Current", "Ex", "Never"))

CT_PMI_75th = st.number_input("Enter CT_PMI_75th(HU)")

gender = st.selectbox("gender", ("Female", "Male"))

BNP = st.selectbox("NT-proBNP/BNP (+)", ("Yes", "No"))

syncope = st.selectbox("syncope", ("Yes", "No"))

DD = st.number_input("Enter D-dimer(mg/L)")

CHF = st.selectbox("CHF", ("Yes", "No"))

chest_pain = st.selectbox("chest pain", ("Yes", "No"))

diabetes = st.selectbox("diabetes", ("Yes", "No"))

CT_PMA_Fat_ratio = st.selectbox("CT_PMA_Fat_ratio", ("<10%", "10%-20%", ">20%"))

CT_PMA_90th = st.number_input("CT_PMA_90th(HU)")

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = model
    
    # Store inputs into dataframe
    X = pd.DataFrame([[gender, BNP, DD, smoke, CHF, diabetes, chest_pain, syncope, oxygen_saturation, CT_PMA_Fat_ratio,
                      CT_PMA_10th, CT_PMA_90th, CT_PMI_75th]], 
                     columns = model.feature_names_in_)
    
    X = X.replace(["Yes", "No"], [1, 0])

    X = X.replace(["Current", "Ex", "Never"], [2,1,0])

    X = X.replace(["<10%", "10%-20%", ">20%"], [0,1,2])

    X = X.replace(["Female", "Male"], [1,0])
    
    # Get prediction
    prediction = clf.predict(X)[0]

    # Output prediction
    if prediction == 1:
        st.text("This patient with pulmonary embolism is expected to require mechanical ventilation")
    else:
        st.text("This patient with pulmonary embolism is not expected to require mechanical ventilation")


    # SHAP解释
    explainer = shap.Explainer(model, X_test[model.feature_names_in_])
    shap_values = explainer.shap_values(X)

    #st.text(f"{shap_values[0]}")
    #st.text(f"{ explainer.expected_value}")
    # 绘制force plot
    fig = shap.force_plot(
            explainer.expected_value, 
            shap_values, 
            feature_names = model.feature_names_in_,
            matplotlib=True,
            show=False)
    

    st.pyplot(fig)

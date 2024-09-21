#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st  
import joblib  
import pandas as pd
import numpy as np


# In[4]:


# app.py  
  
# 加载模型  
  
# 构造模型的完整路径   
with open('random_forest_model.pkl', 'rb') as f:  
    model = pickle.load(f)  
  
# 假设特征名称如下，根据实际情况调整  
feature_names = ['使用呼吸机时间', '体重', 'apache2评分', '喂养途径', '镇静药', '镇痛药', '白蛋白']  
  
def predict(features):  
    # 将特征列表转换为数组  
    features = np.array(features).reshape(1, -1)  
    # 进行预测  
    prediction = model.predict(features)[0]  
    return prediction  
  
def main():  
    st.title("随机森林预测应用")  
  
    # 用户输入特征  
    feature_values = []  
    for feature_name in feature_names:  
        feature_value = st.number_input(f"{feature_name} (输入值):")  
        feature_values.append(feature_value)  
  
    if st.button("进行预测"):  
        prediction = predict(feature_values)  
        st.success(f"预测结果: {prediction}")  
  
if __name__ == "__main__":  
    main()


# In[ ]:





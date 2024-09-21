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
model = joblib.load('random_forest_model.pkl')

  
# 假设特征名称如下，根据实际情况调整  
feature_names = ['使用呼吸机', '体重', 'apache2评分', '喂养途径', '使用镇静剂', '使用镇痛剂', '血清白蛋白']  
  
def predict(features):  
    # 将特征列表转换为数组  
    features = np.array(features).reshape(1, -1)  
    # 进行预测  
    prediction = model.predict(features)[0]  
    return prediction  
  
def main():  
    st.title("随机森林预测应用")
    st.markdown("**说明11111111**")  # 使用Markdown语法添加说明
  
    # 用户输入特征  
    feature_values = []  
    for feature_name in feature_names:  
        feature_value = st.number_input(f"{feature_name} (输入值):")  
        feature_values.append(feature_value)  
  
    if st.button("进行预测"):    
        prediction = predict(feature_values)    
        if prediction == 1:    
            st.success("预测结果: 不耐受")    
        elif prediction == 0:    
            st.success("预测结果: 耐受")   
  
if __name__ == "__main__":  
    main()


# In[ ]:





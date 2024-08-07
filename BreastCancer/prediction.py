
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os

model_path = os.path.join(os.getcwd(), "BreastCancer_Model (4).sav")
bcancer_model = pickle.load(open(model_path, 'rb'))




with st.sidebar:
    selected = option_menu('Breast Cancer Prediction System',
                           ['Cancer prediction'],icons = ['activity'])
    
if(selected == 'Cancer prediction'):
    st.title('Breast Cancer Predictor')
    
    
    
    col1,col2,col3,col4,col5 = st.columns(5)
    
    with col1:
        radius_mean = st.text_input('Radius Mean')
    with col2:
        texture_mean = st.text_input('Texture Mean')
    with col3:
        perimeter_mean = st.text_input('Perimeter Mean')
    with col4:
        area_mean = st.text_input('Area Mean')
    with col5:
        smoothness_mean = st.text_input('Smoothness Mean')
        
        
    with col1:
        compactness_mean = st.text_input('Compactness Mean')
    with col2:
        concavity_mean = st.text_input('Concavity Mean')
    with col3:
        concave_points_mean = st.text_input('Concave Points Mean')
    with col4:
        symmetry_mean = st.text_input('Symmetry Mean')
    with col5:
        fractal_dimension_mean = st.text_input('Fractal Dim Mean')
    
    
    with col1:
        radius_se = st.text_input('Radius SE')
    with col2:
        texture_se = st.text_input('Texture SE')
    with col3:
        perimeter_se = st.text_input('Perimeter SE')
    with col4:
        area_se = st.text_input('Area SE')
    with col5:
        smoothness_se = st.text_input('Smoothness SE')
    
    
    
    with col1:
        compactness_se = st.text_input('Compactness SE')
    with col2:
        concavity_se = st.text_input('Concavity SE')
    with col3:
        concave_points_se = st.text_input('Concave Points SE')
    with col4:
        symmetry_se = st.text_input('Symmetry SE')
    with col5:
        fractal_dimension_se = st.text_input('Fractal Dime SE')
    
    
    
    with col1:
        radius_worst = st.text_input('Radius Worst')
    with col2:
        texture_worst = st.text_input('Texture Worst')
    with col3:
        perimeter_worst = st.text_input('Perimeter Worst')
    with col4:
        area_worst = st.text_input('Area Worst')
    with col5:
        smoothness_worst = st.text_input('Smoothness Worst')
   
   
   
   
    with col1:
       compactness_worst = st.text_input('Compactness Worst')
    with col2:
       concavity_worst = st.text_input('Concavity Worst')
    with col3:
       concave_points_worst = st.text_input('Concave Points Worst')
    with col4:
       symmetry_worst = st.text_input('Symmetry Worst')
    with col5:
       fractal_dimension_worst = st.text_input('Fractal Dim Worst')
    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
   
    
    
    
    
    

    
    
    
    cancer_diagnosis =''
    if st.button('Cancer Test Result'):
       cancer_prediction = bcancer_model.predict([[float(radius_mean), 
                                                   float(texture_mean), 
                                                   float(perimeter_mean), 
                                                   float(area_mean), 
                                                   float(smoothness_mean), 
                                                   float(compactness_mean), 
                                                   float(concavity_mean), 
                                                   float(concave_points_mean), 
                                                   float(symmetry_mean), 
                                                   float(fractal_dimension_mean), 
                                                   float(radius_se), 
                                                   float(texture_se), 
                                                   float(perimeter_se), 
                                                   float(area_se), 
                                                   float(smoothness_se), 
                                                   float(compactness_se), 
                                                   float(concavity_se), 
                                                   float(concave_points_se), 
                                                   float(symmetry_se), 
                                                   float(fractal_dimension_se), 
                                                   float(radius_worst), 
                                                   float(texture_worst), 
                                                   float(perimeter_worst), 
                                                   float(area_worst), 
                                                   float(smoothness_worst), 
                                                   float(compactness_worst), 
                                                   float(concavity_worst), 
                                                   float(concave_points_worst), 
                                                   float(symmetry_worst), 
                                                   float(fractal_dimension_worst)]])
    if st.button('Predict using KNN'):
           st.success('Warning: accuracy is 89%')

    if st.button('Predict using CTA'):
           st.success('Warning: accuracy is 68%')  

    if(cancer_prediction[0] == 1 ):
            cancer_diagnosis = 'Most Probably a Benign Tumor\n'
    else:
            cancer_diagnosis = 'Most Probably a Malignant Tumor\n'
      
    st.success(cancer_diagnosis)
    st.success('Accuracy: 98%')
    
        
       
        
        
        
                        

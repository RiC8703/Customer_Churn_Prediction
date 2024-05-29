from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import predictive_system

def get_input(df_train):
    left, right=st.columns(2)
    with left:
        st.write(f"**Customer's Personal Information**")
        geography=st.radio('To which country does the customer belong?',
                           df_train['Geography'].unique(), horizontal=True, index=1)
        gender=st.radio('What is the gender of the customer?',
                        df_train['Gender'].unique(), horizontal=True)
        age=st.selectbox('What is the age of the customer?',
                         np.arange(18,101))
        credit_score=st.slider("What is customer's credit score?",
                               int(df_train['CreditScore'].min()),
                               int(df_train['CreditScore'].max()),
                               int(df_train['CreditScore'].mean()))
        estimated_salary=st.slider("What is customer's estimated salary?",
                                   0,
                                   200000,
                                   int(df_train['EstimatedSalary'].mean()))
        
    with right:
        st.write(f"**Customer's relationship with the bank.")
        tenure=st.selectbox("What is the customer's tenure with the bank?",
                            sorted(df_train['Tenure'].unique()), index=5)
        balance=st.slider("What is the balance of customer's bank account?",
                          float(df_train['Balance'].min()),
                          float(df_train['Balance'].max()),
                          float(df_train['Balance'].mean()))
        no_of_products=st.radio('How many bank products does customer hold?',
                                sorted(df_train['NumOfProducts'].unique()), horizontal=True, index=1)
        has_credit_card=st.selectbox('Does the customer have a credit card?',
                                     ['Yes', 'No'])
        has_credit_card=1 if has_credit_card=='Yes' else 0
        is_active=st.selectbox('Is the customer actively participating in banking activities?',
                               ['Yes', 'No'])
        is_active=1 if is_active=='Yes' else 0

    X=pd.DataFrame({
        'CreditScore':credit_score,
        'Geography':geography,
        'Gender':gender,
        'Age':age,
        'Tenure':tenure,
        'Balance':balance,
        'NumOfProducts':no_of_products,
        'HasCrCard':has_credit_card,
        'IsActiveMember':is_active,
        'EstimatedSalary':estimated_salary
    }, index=[0])
    return X

if __name__ == '__main__':
    hide_default_format = """
           <style>
           #MainMenu {visibility: hidden; }
           footer {visibility: hidden;}
           </style>
           """
    st.set_page_config(page_title='Bank Customer Churn Predictor', page_icon=None, layout='centered',
                       initial_sidebar_state='auto')
    st.markdown(hide_default_format, unsafe_allow_html=True)

    df_train=pd.read_csv(str(Path(__file__).parents[1] / 'data/Churn_Modelling.csv'))

    st.title('Bank Customer Churn Predictor')

    st.write("""
             This Machine Learning model is used to predict whether a customer would leave tha bank or not. Using the Random Forest Classifier this model will predict if the customer will leave the bank or not based on the input provided by the user.\n 
             Random Forest Classifier was selected for this Machine Learning model beacuse it had the highest f1score among all the other algorithms.
             """)
    
    st.header('Input Fields')
    input_df=get_input(df_train)

    st.header('Prediction')
    customer_churn=predictive_system.predict(input_df)[0]
    if customer_churn==0:
        st.subheader(":green[Customer is not likely to churn]")
    else:
        st.subheader(":red[Customer is likely to churn]")
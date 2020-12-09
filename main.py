import streamlit as st
import pandas as pd
import pickle

# -- Define function to display widgets and store data
def get_input():
    # Display widgets and store their values in variables
    v_Sex = st.sidebar.radio('Sex', ['Male','Female'])
    v_HomeRegion = st.sidebar.radio('HomeRegion', ['Central', 'East', 'North', 'North East', 'South', 'West'])
    v_FacultyName = st.sidebar.radio('FacultyName', ['School of Management', 'School of Sinology', 'School of Law', 'School of Medicine', 'School of Health Science', 'School of Information Technology', 'School of Dentistry',
    'School of Liberal Arts', 'School of Integrative Medicine', 'School of Cosmetic Science', 'School of Nursing', 'School of Agro-industry', 'School of Science', 'School of Social Innovation'])
    # v_Shell_weight = st.sidebar.slider('Shell Weight', 0.0015, 1.0, 0.24)
    # Change the value of sex to be {'M','F','I'} as stored in the trained dataset
    # if v_Sex == 'Male': v_Sex = 'M'
    # elif v_Sex == 'Female': v_Sex = 'F'

    # if v_HomeRegion == 'Central': v_HomeRegion = 1
    # elif v_HomeRegion == 'East': v_HomeRegion = 2

    data = {'Sex': v_Sex,
            'FacultyName': v_FacultyName,
            'HomeRegion': v_HomeRegion}
            # 'Shell_weight': v_Shell_weight}
    # Create a data frame from the above dictionary
    data_df = pd.DataFrame(data, index=[0])
    return data_df
# -- Call function to display widgets and get data from user
df = get_input()

load_sc = pickle.load(open("normalization.pkl", "rb"))
load_knn = pickle.load(open('best_knn.pkl', 'rb'))

data_sample = pd.read_csv('sample.csv')
df = pd.concat([df, data_sample],axis=0)

#HOT CODE
# q_dat = pd.get_dummies(df[['Q29']])
dog_data = pd.get_dummies(df[['HomeRegion']])
fac_data = pd.get_dummies(df[['FacultyName']])

cat_data = pd.get_dummies(df[['Sex']])
X_new = pd.concat([cat_data, fac_data, dog_data, df], axis=1)
# X_new = X_new[:1]
X_new = X_new.drop(columns=['Sex'])
X_new = X_new.drop(columns=['HomeRegion'])
X_new = X_new.drop(columns=['FacultyName'])
# X_new = X_new.drop(columns=['Q29'])

st.header('Actually joinning MFU Prediction:')
st.subheader('User Input:')
st.write(X_new)

prediction = load_knn.predict(X_new)
st.header('Actually joinning MFU Prediction:')
st.subheader('Predicted Output:')
st.write(prediction)

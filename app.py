import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np


df = pd.read_csv('Djerba_Tourist_Activities_Large.csv')
df.head()

df_encoded = pd.get_dummies(df, columns=['Tourist_Type' ,'Interest_Category','Budget','Season','Duration_of_Stay','Accessibility'])
df_encoded.head()
X = df_encoded.drop(columns='Recommended_Activity')
y = df_encoded['Recommended_Activity']

model = DecisionTreeClassifier(random_state=100)
model.fit(X, y)

st.title(' Tourist Activity Recommendation System for Djerba Using Machine Learning')
st.header("Get a Place to visit or Activity Recommendation Based on Your Preferences")

Tourist_Type =st.selectbox("Select your type of group ",["Solo", "Couple", "Family", "Group"])
Interest_Category = st.selectbox('Select your category', ['Relaxation', 'Adventure', 'Culture'])
Budget = st.selectbox('Select your budget', ['Low', 'Medium', 'High'])
Season = st.selectbox('Select your season', ['Spring', 'Fall', 'Summer','Winter'])
Duration_of_Stay = st.selectbox('Select your duration of stay', ['Short', 'Medium','Long'])
Accessibility = st.selectbox('Select your accessibility', ['Yes', 'No'])



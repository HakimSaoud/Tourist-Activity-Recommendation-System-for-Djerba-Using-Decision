import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import numpy as np


df = pd.read_csv('Djerba_Tourist_Activities_Large.csv')


df_encoded = pd.get_dummies(df, columns=['Tourist_Type' ,'Interest_Category'    ,'Budget','Season','Duration_of_Stay','Accessibility'])
df_encoded.head()

X = df_encoded.drop(columns='Recommended_Activity')
y = df_encoded['Recommended_Activity']

model = RandomForestClassifier()
model.fit(X, y)


st.title(' Tourist Activity Recommendation System for Djerba Using Machine Learning')
st.header("Get a Place to visit or Activity Recommendation Based on Your Preferences")

Tourist_Type =st.selectbox("Select your type of group ",["Solo", "Couple", "Family", "Group"])
Interest_Category = st.selectbox('Select your category', ['Relaxation', 'Adventure', 'Culture'])
Budget = st.selectbox('Select your budget', ['Low', 'Medium', 'High'])
Season = st.selectbox('Select your season', ['Spring', 'Fall', 'Summer','Winter'])
Duration_of_Stay = st.selectbox('Select your duration of stay', ['Short', 'Medium','Long'])
Accessibility = st.selectbox('Select your accessibility', ['Walking', 'Public Transport','Car Rental'])



user_input ={
    'Tourist_Type':Tourist_Type,
    'Interest_Category':Interest_Category,
    'Budget':Budget,
    'Season':Season,
    'Duration_of_Stay':Duration_of_Stay,
    'Accessibility':Accessibility
}
user_input_encoded = pd.DataFrame([user_input])
user_input_encoded = pd.get_dummies(user_input_encoded)
user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)



if st.button('Get Recommendation'):
    probabilities = model.predict_proba(user_input_encoded)[0]
    class_labels = model.classes_
    recommendations = pd.DataFrame({'Activity': class_labels, 'Probability': probabilities})
    recommendations = recommendations.sort_values(by='Probability', ascending=False)

    st.subheader("Top Recommendations:")
    for i, row in recommendations.head(3).iterrows():
        st.write(f"- {row['Activity']}")










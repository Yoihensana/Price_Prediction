import streamlit as st
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pickle
import xgboost as xgb

## Load the trained model
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

## Load the encoder
with open('onehot_encoder_loc.pkl', 'rb') as file:
    onehot_encoder_loc = pickle.load(file)


## Streamlit app
st.title('Bangalore Real Estate Price Prediction')

# User Input
Location = st.selectbox('location', onehot_encoder_loc.categories_[0])
Area = st.number_input('area_in_sqft')
Bathroom = st.number_input('bathroom')
BHK = st.number_input('BHK')

if Area / BHK < 350:
    st.write('Invalid entries, please try again with an area per BHK size greater than 350 sqft.')
else:
    if st.button('Predict'):
        # Prepare the input data
        input_data = pd.DataFrame({
            'total_sqft': [Area],
            'bath': [Bathroom],
            'BHK': [BHK]
        })

        # One-hot encode 'Geography'
        loc_encoded = onehot_encoder_loc.transform([[Location]]).toarray()
        loc_encoded_df = pd.DataFrame(loc_encoded, columns=onehot_encoder_loc.get_feature_names_out(['location']))

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), loc_encoded_df], axis=1)

        # Predict price
        prediction = model.predict(input_data)

        st.write(f'Price: {prediction[0]:.2f} Lakhs')



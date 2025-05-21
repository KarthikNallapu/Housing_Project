import streamlit as st
import pandas as pd
from joblib import load  # ✅ Use joblib instead of pickle

# Load model and data
@st.cache_resource
def load_model():
    return load('hyderabad_housing_model.pkl')  # ✅ Load using joblib

@st.cache_data
def load_data():
    df = pd.read_csv('Hyderabad.csv')
    df.columns = df.columns.str.strip()
    return df

def main():
    st.title('Hyderabad House Price Prediction')
    df = load_data()
    model = load_model()

    # Features for user input
    locations = sorted(df['Location'].unique())
    amenities = [
        'MaintenanceStaff','Gymnasium','SwimmingPool','LandscapedGardens','JoggingTrack',
        'RainWaterHarvesting','IndoorGames','ShoppingMall','Intercom','SportsFacility','ATM',
        'ClubHouse','School','24X7Security','PowerBackup','CarParking','StaffQuarter','Cafeteria',
        'MultipurposeRoom','Hospital','WashingMachine','Gasconnection','AC','Wifi',
        "Children'splayarea",'LiftAvailable','BED','VaastuCompliant','Microwave','GolfCourse','TV',
        'DiningTable','Sofa','Wardrobe','Refrigerator'
    ]

    with st.form("input_features"):
        st.header("Property Details")
        location = st.selectbox("Location", locations)
        area = st.number_input("Area (sqft)", min_value=200, max_value=10000, value=1200)
        bedrooms = st.slider("No. of Bedrooms", 1, 10, 2)
        resale = st.radio("Resale Property", [0, 1], format_func=lambda x: "Yes" if x else "No")
        selected_amenities = st.multiselect("Amenities", amenities)
        submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Prepare input data
        input_data = {
            'Area': area,
            'Location': location,
            'No. of Bedrooms': bedrooms,
            'Resale': resale,
        }

        # Add amenity flags
        for amenity in amenities:
            input_data[amenity] = 1 if amenity in selected_amenities else 0

        # Add total amenities count
        input_data['TotalAmenities'] = len(selected_amenities)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Price: ₹{prediction:,.0f}")
            st.write("**Input Summary:**")
            st.json(input_data)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == '__main__':
    main()

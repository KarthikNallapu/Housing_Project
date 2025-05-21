# 🏠 Hyderabad Housing Price Prediction

This project is a **machine learning-based web application** built to predict house prices in Hyderabad based on user inputs including area, location, number of bedrooms, resale status, and various amenities.

The project uses:
- **Random Forest Regressor** for price prediction.
- **Streamlit** for building the user interface.
- **Pandas**, **scikit-learn**, and **matplotlib** for data preprocessing, modeling, and visualization.

---

## 📂 Project Structure

Housing_Project/
│
├── Hyderabad.csv # Dataset with housing information
├── hyderabad_housing_model.pkl # Trained ML model
├── feature_importance.png # Top features influencing the prediction
├── housing_app.py # Streamlit app code
└── README.md # Project documentation


---

## 🚀 Features

- Preprocessing pipeline using `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.
- Training a `RandomForestRegressor` model using `GridSearchCV`.
- Amenities handling with automatic feature generation (`TotalAmenities`).
- Top 15 important features visualized with a bar plot.
- Interactive web app with:
  - Location selection
  - Area input
  - Bedrooms slider
  - Resale toggle
  - Amenity checklist
- Model predictions shown in a user-friendly format.

---

## 🧠 Technologies Used

| Task               | Libraries/Tools Used              |
|--------------------|----------------------------------|
| Data Processing    | `pandas`, `numpy`                |
| ML Modeling        | `scikit-learn`, `RandomForest`   |
| Web Interface      | `streamlit`                      |
| Model Saving       | `pickle`                         |
| Visualization      | `matplotlib`                     |

---

## 🏁 Getting Started
### 1. Clone the Repository
git clone https://github.com/KarthikNallapu/Housing_Project.git
cd Housing_Project

### 2. Install Requirements
Make sure you have Python installed (recommended: Python 3.8+)

bash
pip install -r requirements.txt
Note: If requirements.txt is not present, manually install:
pip install pandas numpy scikit-learn matplotlib streamlit

### 3. Run the Streamlit App
streamlit run housing_app.py
The app will open in your browser at http://localhost:8501/.

### 📊 Dataset
The dataset Hyderabad.csv includes:

Basic features: Area, No. of Bedrooms, Resale, Location
Multiple binary features for amenities
Price: Target variable

### 🔍 Model Evaluation
After training, the model was evaluated using:

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² Score

Example output:
MAE: ₹75,321.56
RMSE: ₹104,239.45
R2: 0.8423

### 💾 Model Saving
The trained pipeline is saved using Python pickle for later use:

with open('hyderabad_housing_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

### 🙌 Acknowledgments
Scikit-learn
Streamlit
matplotlib
Open-source housing data used for educational purposes.

### 📬 Contact
Karthik Nallapu
GitHub: @KarthikNallapu






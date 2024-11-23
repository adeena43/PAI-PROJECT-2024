import streamlit as st
import numpy as np
from joblib import load

# Load the trained model
model = load('Boston.joblib')

# Set custom CSS for styling (including gradient background and image styling)
st.markdown("""
    <style>
        body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100vh; /* Full viewport height */
            font-family: 'Arial', sans-serif;
            background-color: #00203FFF ; /* Blue background color */
            display: flex;
            justify-content: center; /* Center content horizontally */
            align-items: center; /* Center content vertically */
            flex-direction: column; /* Stack content vertically */
            text-align: center; /* Center align text */
        }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: #00203FFF; /* Background color */
            width: 100%;
            height: 100vh;
        }

        .block-container {
            width: 100%; /* Full width */
            max-width: 1200px; /* Optional: Limit maximum width for better readability */
            text-align: center; /* Center align content */
        }

        .stImage {
            display: block; /* Make the image a block-level element */
            width: 100%;  /* Full width */
            height: auto; /* Maintain aspect ratio */
            object-fit: cover; /* Ensures the image maintains aspect ratio and fits */
            margin-left: auto; /* Center horizontally */
            margin-right: auto; /* Center horizontally */
        }

        /* Sidebar Styling */
        .stSidebar {
            background-color: #ADEFD1FF; /* Light green background for sidebar */
            color: white; /* Default text color inside the sidebar */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.1);
        }

        /* Sidebar Title */
        .stSidebar h1 {
            font-size: 22px;
            font-weight: bold;
            color: #00203FFF; /* Dark blue text for the sidebar title */
            text-align: center;
        }

        /* Sidebar Subtext */
        .stSidebar p {
            color: red; /* Red text for the subtext */
            font-size: 16px;
            text-align: center;
        }

        .stTextInput, .stNumberInput, .stSelectbox {
            width: 100%;  /* Full width for input fields */
            max-width: 450px; /* Limit input field width */
            background-color: rgba(255, 255, 255, 0.9); /* Light transparent background */
            border-radius: 10px;
            font-size: 16px;
            padding: 12px;
            margin-bottom: 15px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
            transition: 0.3s ease-in-out;
        }

        .stTextInput:focus, .stNumberInput:focus, .stSelectbox:focus {
            outline: none;
            border: 2px solid #4CAF50; /* Green focus border */
            box-shadow: 0 0 5px #4CAF50;
        }

        .stButton>button {
            width: 100%;  /* Full width button */
            max-width: 250px; /* Optional: Limit button width */
            background-color: #ADEFD1FF;
            color: red;
            font-size: 18px;
            border-radius: 12px;
            padding: 15px;
            margin-top: 20px; /* Add space above the button */
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #4CAF50;
            color: white;
        }

        .stMarkdown {
            color: white; /* Ensure text contrasts with the blue background */
            font-size: 24px;
            font-weight: bold;
        }

        /* Additional Input Styles for Placeholders */
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            color: #333; /* Darker text color */
            font-weight: normal;
        }

        .stTextInput input::placeholder, .stNumberInput input::placeholder {
            color: #888; /* Lighter placeholder text */
        }
    </style>
""", unsafe_allow_html=True)

# Add an image of a house with "BOSTON HOUSING SCHEME" text as the header
st.image("my_img.jpg", caption="Boston Housing Scheme", use_container_width=True)

# Title with custom formatting
st.markdown("<h1 style='text-align: center; color: white;'>BOSTON HOUSING SCHEME</h1>", unsafe_allow_html=True)
st.markdown("""
    <h3 style='text-align: center; color: #D6ED17FF;'>
        Enter the features below and get the predicted price of the house
    </h3>
""", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.title("HOUSING FEATURES")
st.sidebar.write("<p>Please enter the features below:</p>", unsafe_allow_html=True)

# Feature inputs using text fields for manual input with placeholders
CRIM = st.sidebar.text_input("Per capita crime rate (CRIM)", value="0.0", placeholder="Enter a numeric value")
ZN = st.sidebar.text_input("Proportion of residential land zoned for lots over 25,000 sq.ft. (ZN)", value="0.0", placeholder="Enter a numeric value")
INDUS = st.sidebar.text_input("Proportion of non-retail business acres per town (INDUS)", value="0.0", placeholder="Enter a numeric value")
CHAS = st.sidebar.selectbox("Charles River dummy variable (CHAS)", [0, 1])
NOX = st.sidebar.text_input("Nitric oxide concentration (NOX)", value="0.5", placeholder="Enter a numeric value")
RM = st.sidebar.text_input("Average number of rooms per dwelling (RM)", value="5.0", placeholder="Enter a numeric value")
AGE = st.sidebar.text_input("Proportion of owner-occupied units built prior to 1940 (AGE)", value="50.0", placeholder="Enter a numeric value")
DIS = st.sidebar.text_input("Weighted distances to five Boston employment centres (DIS)", value="5.0", placeholder="Enter a numeric value")
RAD = st.sidebar.text_input("Index of accessibility to radial highways (RAD)", value="1.0", placeholder="Enter a numeric value")
TAX = st.sidebar.text_input("Full-value property-tax rate per $10,000 (TAX)", value="200.0", placeholder="Enter a numeric value")
PTRATIO = st.sidebar.text_input("Pupil-teacher ratio by town (PTRATIO)", value="15.0", placeholder="Enter a numeric value")
B = st.sidebar.text_input("Proportion of blacks by town (B)", value="300.0", placeholder="Enter a numeric value")
LSTAT = st.sidebar.text_input("% lower status of the population (LSTAT)", value="12.0", placeholder="Enter a numeric value")

# Convert input to a numpy array
features = np.array([[float(CRIM), float(ZN), float(INDUS), CHAS, float(NOX), float(RM), float(AGE), float(DIS), float(RAD), float(TAX), float(PTRATIO), float(B), float(LSTAT)]])

# Debug feature input
st.write("Input Features:", features)

# Predict button
if st.button("Predict 🏠"):
    prediction = model.predict(features)
    
    # Display the predicted price in bold and bigger format
    st.markdown(f"<h2 style='text-align: center; color: #D6ED17FF; font-weight: bold;'>Predicted House Price</h2>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>${prediction[0]:,.2f}</h1>", unsafe_allow_html=True)

    # Celebrate with balloons!
    st.balloons()


"""Streamlit version of StudyHourPredictor (Minimal Version)"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
import streamlit as st



def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = f"data:image/png;base64,{data.encode('base64')}" if hasattr(data, 'encode') else f"data:image/png;base64,{__import__('base64').b64encode(data).decode()}"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
<style>
div[data-baseweb="slider"] {
    width: 56% !important;
}
</style>
""", unsafe_allow_html=True)



add_bg_from_local("back.png")


warnings.filterwarnings("ignore", message="X does not have valid feature names")


st.title("📚 Study Hour Predictor")
st.write("This app predicts your **Focus Score**, **Productivity Score**, and **Best Study Time** based on your daily habits.")


df = pd.read_csv("study hour.csv")


X = df[['Sleep_Hours', 'Mobile_Use_Hours', 'Study_Time_Hours']]
y_focus = df['Focus_Score']
y_productivity = df['Productivity_Score']
y_time = df['Study_Time']


X_train_r1, X_test_r1, y_train_r1, y_test_r1 = train_test_split(X, y_focus, test_size=0.2, random_state=42)
X_train_r2, X_test_r2, y_train_r2, y_test_r2 = train_test_split(X, y_productivity, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_time, test_size=0.2, random_state=42)


focus_model = LinearRegression()
focus_model.fit(X_train_r1, y_train_r1)

productivity_model = LinearRegression()
productivity_model.fit(X_train_r2, y_train_r2)

time_model = LogisticRegression(max_iter=1000)
time_model.fit(X_train_c, y_train_c)

st.subheader("Make Predictions")



Sleep_Hours = float(st.text_input("😴 Hours of Sleep :", "7"))
Mobile_Use_Hours = float(st.text_input("📱 Hours of Mobile Use :", "3"))
Study_Time_Hours = float(st.text_input("📖 Hours of Study :", "4"))


if st.button("Predict"):

    total_hours = Sleep_Hours + Mobile_Use_Hours + Study_Time_Hours  # now numeric

    if total_hours > 24:
        st.warning("⚠️ The total of Sleep, Mobile Use, and Study hours cannot exceed 24 hours!")
    else:
        features = [[Sleep_Hours, Mobile_Use_Hours, Study_Time_Hours]]

        pred_focus = float(focus_model.predict(features)[0])
        pred_productivity = float(productivity_model.predict(features)[0])
        pred_focus = max(0, min(pred_focus, 10))
        pred_productivity = max(0, min(pred_productivity, 100))

        pred_time = time_model.predict(features)[0]
        time_dict = {1: "Morning", 2: "Evening", 3: "Night"}

        
        st.success(f"Predicted Focus Score: {pred_focus:.2f}")
        st.success(f"Predicted Productivity Score: {pred_productivity:.2f}")
        st.info(f"Suggested Best Study Time: {time_dict.get(pred_time, 'Unknown')}")



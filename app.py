import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="✈️ Flight Fare Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center; color: #FFA500;'>✈️ Flight Fare Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("Select Model")
model_files = [f for f in os.listdir("models") if f.endswith("_best.pkl")]
model_names = [f.replace("_best.pkl","") for f in model_files]
selected_model_name = st.sidebar.selectbox("Choose a model", model_names)
model_path = f"models/{selected_model_name}_best.pkl"
model = joblib.load(model_path)

st.sidebar.markdown("---")

st.subheader("Flight Details")
with st.expander("Enter Flight Details"):
    col1, col2, col3 = st.columns(3)

    with col1:
        total_stops = st.slider("Total Stops ✈️", 0, 4, 1)
        dep_hour = st.slider("Departure Hour ⏰", 0, 23, 10)
        dep_min = st.slider("Departure Minute ⏰", 0, 59, 30)
        day_of_week = st.selectbox("Day of Week 📅", list(range(7)),
                                   format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

    with col2:
        arrival_hour = st.slider("Arrival Hour ⏰", 0, 23, 15)
        arrival_min = st.slider("Arrival Minute ⏰", 0, 59, 45)
        month = st.selectbox("Month 🗓️", list(range(1,13)))
        airline = st.selectbox("Airline 🛫", ["Air India", "GoAir", "IndiGo", "Jet Airways", 
                                            "Jet Airways Business", "Multiple carriers", 
                                            "Multiple carriers Premium economy", "SpiceJet",
                                            "Trujet", "Vistara", "Vistara Premium economy"])

    with col3:
        source = st.selectbox("Source 📍", ["Chennai", "Kolkata", "Delhi", "Mumbai", "Bangalore"])
        destination = st.selectbox("Destination 📍", ["Cochin", "Delhi", "Hyderabad", "Kolkata", "New Delhi"])

dep_time = dep_hour*60 + dep_min
arrival_time = arrival_hour*60 + arrival_min
duration = arrival_time - dep_time
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)


def get_pipeline_feature_names(pipeline):
    ct = None
    for step_name, step in pipeline.named_steps.items():
        if hasattr(step, "transformers_") or hasattr(step, "get_feature_names_out"):
            ct = step
            break

    if ct is None:

        return pipeline.named_steps['model'].feature_names_in_

    output_features = []
    if hasattr(ct, "transformers_"):  
        for name, transformer, cols in ct.transformers_:
            if transformer == 'passthrough':
                output_features.extend(cols)
            elif hasattr(transformer, 'get_feature_names_out'):
                output_features.extend(transformer.get_feature_names_out(cols))
            else:
                output_features.extend(cols)
    elif hasattr(ct, "get_feature_names_out"):
        output_features.extend(ct.get_feature_names_out())
    return output_features

feature_names = get_pipeline_feature_names(model)

input_dict = dict.fromkeys(feature_names, 0)

numeric_features = {
    "Total_Stops": total_stops,
    "Dep_time": dep_time,
    "Arrival_time": arrival_time,
    "Duration": duration,
    "Day_of_week": day_of_week,
    "Month_sin": month_sin,
    "Month_cos": month_cos
}
input_dict.update(numeric_features)

def set_categorical(feature_prefix, value):
    matches = [f for f in feature_names if f.lower() == f"{feature_prefix}_{value}".lower()]
    if matches:
        input_dict[matches[0]] = 1

for prefix, val in [("Airline", airline), ("Source", source), ("Destination", destination)]:
    set_categorical(prefix, val)

input_df = pd.DataFrame([input_dict])



st.markdown("---")
st.subheader("Prediction Result")
if st.button("Predict Price 💰", use_container_width=True):
    try:
        price_pred = model.predict(input_df)[0]
        st.metric(label="Estimated Flight Price", value=f" {price_pred:.2f}")

        # Show a small price range
        lower = price_pred * 0.9
        upper = price_pred * 1.1
        st.info(f"Estimated Price Range:  {lower:.2f} -  {upper:.2f}")

        # Show summary of flight details
        st.markdown("### Flight Details Summary")
        st.write({
            "Total Stops": total_stops,
            "Departure Time": f"{dep_hour:02d}:{dep_min:02d}",
            "Arrival Time": f"{arrival_hour:02d}:{arrival_min:02d}",
            "Duration (min)": duration,
            "Day of Week": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day_of_week],
            "Month": month,
            "Airline": airline,
            "Source": source,
            "Destination": destination
        })

    except Exception as e:
        st.error(f"Prediction failed: {e}")


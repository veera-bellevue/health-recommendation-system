import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime
from openai import OpenAI
#Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

#load Model ID from config
FINE_TUNED_MODEL = st.secrets["FINE_TUNED_MODEL"]

#Set page configuration
st.set_page_config(
    page_title = "Health Recommendation System",
    page_icon = "❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Define health profile structure
def create_empty_profile():
    return {
        "age_group": "",
        "gender":"",
        "bmi":0.0,
        "bmi_category":"",
        "activity_level":"",
        "diet_category":"",
        "risk_category":"",
        "medication_Category":"",
        "medication_count":0,
    }
#Get recommendation from fine-tuned model
def get_recommendation(profile):
    client = get_openai_client()
    #Fomrat profile into context
    context = f"""Patient Profile:
-Demographics: {profile['age_group']}, {profile['gender']}
-Body Measurements: BMI {profile['bmi']:.1f}, ({profile['bmi_category']})
-Physical Activity: {profile['activity_level']}
-Diet Quality: {profile['diet_category']}
-Health Risks: {profile['risk_category']}
-Medications: {profile['medication_category']} complexity ({profile['medication_count']} medications)"""
    try:
        response = client.chat.completions.create(
            model = FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": "You are a personalized health and wellness advisor. Provide evidence-based recommendations based on the user's health profile."},
                {"role": "user", "content": context}
            ],
            temperature = 0.7,
            max_tokens = 500
        )
        recommendation = response.choices[0].message.content

        #Log recommendation
        log_recommendation(profile, recommendation)
        return recommendation
    except Exception as e:
        st.error(f"Error generating recommendation: {str(e)}")
        return None

#Log recommendation to file
def log_recommendation(profile, recommendation):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'profile': profile,
        'recommendation': recommendation
    }

    try:
        # Load existing  logs
        try:
            with open('recommendation_log.jsonl', 'r') as f:
                logs = [json.loads(line) for line in f]
        except:
            logs = []
        #Append new log
        logs.append(log_entry)

        #Save logs (keep last 1000 only)
        with open('recommendation_log.jsonl', 'w') as f:
            for log in logs[-1000:]:
                f.write(json.dumps(log) + '\n')
    except Exception as e:
        st.warning(f"Could not log recommendation: {str(e)}")

#Calculate BMI and category 
def calculate_bmi(weight, height):
    #Height in meters, weight in kg
    bmi = weight / ((height/100) ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return round(bmi, 1), category
#Main app 
def main():
    #Sidebar
    st.sidebar.title("Health Recommendation System")
    st.sidebar.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    page = st.sidebar.radio("Navigation", ["Home", "Get Recommendations", "View Analytics"])

    if page == "Home":
        display_home()
    elif page == "Get Recommendations":
        display_recommendation_form()
    elif page == "View Analytics":
        display_analytics()

#Home page
def display_home():
    st.title("Welcome to the Health Recommendation System")

    st.markdown("""
    This application provides personalized health and wellness recommendations
    based on your health profile. It uses a model fine-tuned on comprehensive
    health data from the National Health and Nutrition Examination Survey (NHANES).

    ### How it works:
    1. Enter your health information
    2. Receive personalized recommendations
    3. Track your progress over time

    ### Benefits:
    - Evidence-based recommendations
    - Tailored to your specific health profile
    - Privacy-focused (your data stays on your device)

    ### Get Started
    Click on "Get Recommendations" in the sidebar to begin.
    """)
    st.image("https://img.icons8.com/color/96/000000//health.png", width=100)

#Recommendation form
def display_recommendation_form():
    st.title("Your Personal Health Recommendations")

    #Create form for user inputs
    with st.form("health_profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Information")
            age_group = st.selectbox(
            "Age Group",
            ["Under 18", "18-35", "36-50", "51-65", "Over 65"]
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            height = st.number_input(
                "Height (cm)",
                min_value = 100.0,
                max_value = 250.0,
                value=170.0,
                step=1.0
            )
            
            weight = st.number_input(
                "Weight (kg)",
                min_value = 30.0,
                max_value = 250.0,
                value=70.0,
                step=1.0
            )

            med_count = st.number_input(
                "Number of Medications",
                min_value = 0,
                max_value = 20,
                value=0,
                step=1
            )
        with col2:
            st.subheader("Health and Lifestyle")

            activity_level = st.select_slider(
                "Physical Activity Level:",
                options=["Low", "Moderate", "High"],
                value="Moderate"
            )

            diet_quality = st.select_slider(
                "Diet Quality",
                options=["Poor", "Needs Improvement", "Good", "Excellent"],
                value="Good"
            )

            health_conditions = st.multiselect(
                "Health Conditions",
                ["None", "Hypertension", "Diabetes", "High Cholesterol", "Heart Disease", "Asthma", "Arthritis", "Depression", "Anxiety", "Other"],
                default=["None"]
            )

            if "None" in health_conditions and len(health_conditions) > 1:
                health_conditions.remove("None")

            #Determine risk category based on conditions
            if len(health_conditions) == 0 or (len(health_conditions) == 1 and health_conditions[0] == "None"):
                risk_category = "Low Risk"
            elif len(health_conditions) <= 2:
                risk_category = "Moderate Risk"
            else:
                risk_category = "High Risk"

            #Determine medication complexity
            if med_count == 0:
                med_category = "Low"
            elif med_count <= 3:
                med_category = "Moderate"
            else:
                med_category = "High"
        submit_button = st.form_submit_button("Get Recommendations")
    #Calculate BMI and show profile
    if submit_button:
        bmi, bmi_category = calculate_bmi(weight, height)

        #Create health profile
        profile = {
            "age_group": age_group,
            "gender": gender,
            "bmi": bmi,
            "bmi_category": bmi_category,
            "activity_level": activity_level,
            "diet_category": diet_quality,
            "risk_category": risk_category,
            "medication_category": med_category,
            "medication_count": med_count
        }

        #Display profile summary
        st.subheader("Your Health Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("BMI", f"{bmi}", f"{bmi_category}")
            st.metric("Age Group", profile["age_group"])

        with col2:
            st.metric("Activity Level", profile["activity_level"])
            st.metric("Diet Quality", profile["diet_category"])

        with col3:
            st.metric("Health Risk", profile["risk_category"])
            st.metric("Medications", f"{profile['medication_count']} ({profile['medication_category']})")

        #Get and display recommendation
        with st.spinner("Generating your personalized recommendations..."):
            recommendation = get_recommendation(profile)
        if recommendation:
            st.success("Your personalized recommendations are ready!")
            st.subheader("Recommendations")
            st.write(recommendation)

            #Optional: Save recommendations
            if st.button("Save Recommendations"):
                try:
                    with open(f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "W") as f:
                        f.write(f"Health Profile:\n")
                        for key, value in profile.items():
                            f.write(f"{key}: {value}\n")
                        f.write(f"\nRecommendations:\n{recommendation}\n")
                    st.success("Recommendations saved!")
                except Exception as e:
                    st.error(f"Error saving recommendations: {str(e)}")
                    
#Analytics page
def display_analytics():
    st.title("Analytics and Insights")

    try:
        # Load recommendation logs
        try:
            # First check if the file exists
            if not os.path.exists('recommendation_log.jsonl'):
                st.info("No recommendation data available yet. Please generate some recommendations first.")
                return

            # Then try to load the file
            logs = []
            with open('recommendation_log.jsonl', 'r') as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        st.warning(f"Skipped invalid JSON line in logs")

            if not logs:
                st.info("No recommendation data available yet.")
                return

            # Create DataFrame from logs
            df = pd.DataFrame(logs)

            # Extract profile data into columns - handle case where profile might be a string
            profile_cols = [
                'age_group', 'gender', 'bmi_category',
                'activity_level', 'diet_category', 'risk_category',
                'medication_category'
            ]

            # Create empty columns first
            for col in profile_cols:
                df[col] = None

            # Now populate them safely
            for i, row in df.iterrows():
                try:
                    profile = row['profile']
                    if isinstance(profile, str):
                        profile = json.loads(profile)

                    for col in profile_cols:
                        if col in profile:
                            df.at[i, col] = profile[col]
                except (KeyError, TypeError, json.JSONDecodeError):
                    # Skip this row if there's an issue
                    continue

            # Show summary statistics
            st.subheader("Summary")
            st.write(f"Total recommendations: {len(df)}")

            col1, col2 = st.columns(2)

            # Only create visualizations if we have enough data
            if len(df) > 0:
                with col1:
                    # Age group distribution
                    if df['age_group'].notna().any():
                        st.subheader("Age Group Distribution")
                        age_counts = df['age_group'].value_counts()
                        fig1, ax1 = plt.subplots()
                        ax1.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%')
                        ax1.axis('equal')
                        st.pyplot(fig1)

                    # Activity level distribution
                    if df['activity_level'].notna().any():
                        st.subheader("Activity Level Distribution")
                        activity_counts = df['activity_level'].value_counts()
                        fig3, ax3 = plt.subplots()
                        sns.barplot(x=activity_counts.index, y=activity_counts.values, ax=ax3)
                        st.pyplot(fig3)

                with col2:
                    # BMI category distribution
                    if df['bmi_category'].notna().any():
                        st.subheader("BMI Category Distribution")
                        bmi_counts = df['bmi_category'].value_counts()
                        fig2, ax2 = plt.subplots()
                        sns.barplot(x=bmi_counts.index, y=bmi_counts.values, ax=ax2)
                        plt.xticks(rotation=45)
                        st.pyplot(fig2)

                    # Health risk distribution
                    if df['risk_category'].notna().any():
                        st.subheader("Health Risk Distribution")
                        risk_counts = df['risk_category'].value_counts()
                        fig4, ax4 = plt.subplots()
                        sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=ax4)
                        st.pyplot(fig4)
            else:
                st.info("Not enough data to generate visualizations.")

        except Exception as e:
            st.error(f"Error loading recommendation logs: {str(e)}")
            st.exception(e)  # This will show the full traceback in development
            return

    except Exception as e:
        st.error(f"Error in analytics: {str(e)}")
        st.exception(e)
if __name__ == "__main__":
    main()
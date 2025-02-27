Health Recommendation System
A personalized health and wellness recommendation system built using Streamlit and OpenAI's fine-tuned GPT model. The application provides tailored health recommendations based on individual health profiles, leveraging data from the National Health and Nutrition Examination Survey (NHANES).
Features

Health Profile Input: Enter your demographic info, body measurements, activity level, and medical history
Personalized Recommendations: Receive evidence-based health and wellness advice tailored to your profile
Analytics Dashboard: View statistics and patterns of health profiles and recommendations
AI-Powered: Utilizes a fine-tuned OpenAI model trained on NHANES health data

Installation

Clone this repository:
git clone https://github.com/yourusername/health-recommendation-system.git
cd health-recommendation-system

Install required packages:
pip install -r requirements.txt

Set up the secrets file:
mkdir -p .streamlit
cp secrets.toml.example .streamlit/secrets.toml

Edit .streamlit/secrets.toml to include your OpenAI API key and fine-tuned model ID:

CopyOPENAI_API_KEY = "your-openai-api-key"
FINE_TUNED_MODEL = "your-fine-tuned-model-id"

Usage

Run the Streamlit app:

bashCopystreamlit run app.py

Open your web browser to http://localhost:8501
Navigate to the "Get Recommendations" page to enter your health profile
View your personalized recommendations

Project Structure

app.py: Main Streamlit application
requirements.txt: Python dependencies
.streamlit/config.toml: Streamlit configuration
.streamlit/secrets.toml: API keys and model IDs (not included in repo)
secrets.toml.example: Template for the secrets file

Model Training
The recommendation system uses a fine-tuned GPT-3.5-Turbo model trained on health data from NHANES (National Health and Nutrition Examination Survey). The model was trained to provide personalized recommendations based on:

Demographics (age, gender)
Body measurements (BMI, weight, height)
Physical activity levels
Dietary habits
Medical conditions
Medication usage

Customization
You can modify the application to include additional health metrics or change the recommendation categories:

Update the create_empty_profile() function to include new health parameters
Modify the form in display_recommendation_form() to capture additional data
Update the context string in get_recommendation() to include the new parameters

Security Note
This application requires an OpenAI API key. Never commit your API keys or secrets to GitHub. The .streamlit/secrets.toml file is included in .gitignore to prevent accidental exposure of sensitive information.
Future Enhancements

Integration with wearable device data
Longitudinal tracking of health metrics
More detailed nutrition and exercise plans
Mobile application development

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

NHANES for providing comprehensive health data
OpenAI for the fine-tuning capabilities
Streamlit for the web application framework

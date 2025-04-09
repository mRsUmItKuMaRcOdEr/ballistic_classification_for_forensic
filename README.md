# Ballistic Classification for Forensic Investigation

This application uses machine learning to predict damage levels based on ballistic characteristics. It analyzes various parameters such as caliber, velocity, mass, and more to provide accurate damage assessments.

## Deployment on Streamlit Cloud

To deploy this app on Streamlit Cloud:

1. Create a GitHub repository and push this code to it
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file path (ballistics_app.py)
6. Click "Deploy"

## Local Development

To run the app locally:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run ballistics_app.py
   ```

## Requirements

The app requires the following packages:
- streamlit
- joblib
- pandas
- numpy
- plotly
- scikit-learn
- matplotlib
- seaborn

All dependencies are listed in the requirements.txt file. 

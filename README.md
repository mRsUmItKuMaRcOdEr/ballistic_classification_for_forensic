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

### Troubleshooting Deployment Issues

If you encounter deployment issues:

1. **Package Compatibility**: The app uses specific package versions that are compatible with Streamlit Cloud. If you update packages, make sure to test locally first.

2. **Model Files**: The app includes a demo mode that works without model files. If you want to use the full version with the trained model, you'll need to:
   - Train the model locally
   - Upload the model files (best_model.joblib, preprocessor.joblib, feature_selector.joblib) to your Streamlit Cloud deployment

3. **Python Version**: Streamlit Cloud uses Python 3.12.9. If you're developing locally with a different version, you might encounter compatibility issues.

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

All dependencies are listed in the requirements.txt file with specific versions that are compatible with Streamlit Cloud. 

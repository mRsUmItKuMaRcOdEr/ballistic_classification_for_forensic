import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def create_streamlit_app(model, preprocessor, class_names, feature_names):
    """Streamlit app with ALL original categories preserved"""
    # Set dark theme
    st.set_page_config(page_title="Ballistic Forensics", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for better styling with dark theme
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #E0E0E0;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: rgba(30, 34, 45, 0.9);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .highlight {
        background-color: rgba(30, 136, 229, 0.2);
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    /* Dark theme modifications */
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .element-container {
        background-color: transparent;
    }
    .css-1544g2n {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMarkdown, .stDataFrame {
        background-color: rgba(30, 34, 45, 0.9);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30, 34, 45, 0.9);
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #E0E0E0;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: transparent;
    }
    div[data-testid="stExpander"] {
        background-color: rgba(30, 34, 45, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
    }
    .streamlit-expanderHeader {
        color: #E0E0E0 !important;
    }
    /* Hide empty markdown blocks */
    .element-container:empty {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with custom styling
    st.markdown('<h1 class="main-header">🔫 Ballistic Data Classification for Forensic Investigation</h1>', unsafe_allow_html=True)
    
    # Introduction section with dark theme
    st.markdown("""
    <div class="info-box">
    <h3 style="color: #E0E0E0;">About This Application</h3>
    <p style="color: #E0E0E0;">This application uses machine learning to predict damage levels based on ballistic characteristics. 
    It analyzes various parameters such as caliber, velocity, mass, and more to provide accurate damage assessments.</p>
    <p style="color: #E0E0E0;">Simply adjust the parameters in the sidebar and click 'Analyze Damage' to get your prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== Input Section =====
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Firearm Parameters</h2>', unsafe_allow_html=True)
        
        # Create a container for the input parameters
        input_container = st.container()
        
        with input_container:
            # Numeric inputs (2 columns for better layout)
            col1, col2 = st.columns(2)
            with col1:
                caliber = st.slider('Caliber (mm)', 5.0, 15.0, 9.0, 0.1)
                velocity = st.slider('Velocity (m/s)', 300.0, 1000.0, 500.0)
            with col2:
                mass = st.slider('Mass (g)', 2.0, 20.0, 8.0, 0.1)
                distance = st.slider('Distance (m)', 1, 100, 20)
            
            rifling_twist = st.slider('Rifling Twist Rate', 6.0, 14.0, 10.0, 0.1)

            # Categorical features
            bullet_type = st.selectbox('Bullet Type', [
                'FMJ', 'JHP', 'SP', 'AP', 'BT', 'V-MAX', 'ELD-M', 'AMAX',
                'BTHP', 'TMJ', 'EFMJ', 'Critical Defense', 'Critical Duty',
                'Golden Saber', 'Hydra-Shok', 'Ranger T-Series', 'HST',
                'V-Crown', 'Gold Dot', 'Black Talon', 'PolyCase ARX',
                'Extreme Point', 'PowerBond', 'AccuBond', 'Partition'
            ])
        
            gun_type = st.selectbox('Gun Type', [
                'Glock 17', 'Beretta 92FS', 'Sig Sauer P320', 'Colt 1911', 
                'Smith & Wesson M&P', 'HK USP', 'CZ 75', 'FN Five-seveN',
                'Desert Eagle', 'Ruger SR9', 'Walther PPQ', 'Springfield XD',
                'AK-47', 'AR-15', 'M16', 'M4 Carbine', 'HK416', 'SCAR-L',
                'Remington 700', 'Winchester Model 70', 'Barrett M82', 'Dragunov SVD',
                'Mossberg 500', 'Remington 870', 'Benelli M4', 'Kel-Tec KSG'
            ])

            target_material = st.selectbox('Target Material', [
                'Wood', 'Metal', 'Glass', 'Concrete', 'Kevlar', 
                'Ceramic', 'Rubber', 'Water', 'Sand', 'Brick'
            ])
        
        # Add a divider
        st.markdown("---")
        
        # Add information about the model in a container
        with st.container():
            st.markdown("""
            <div class="info-box">
            <h4 style="color: #E0E0E0;">Model Information</h4>
            <p style="color: #E0E0E0;">This application uses a machine learning model trained on ballistic data to predict damage levels.</p>
            <p style="color: #E0E0E0;">Damage levels are classified as:</p>
            <ul style="color: #E0E0E0;">
                <li>Negative: Minimal or no damage</li>
                <li>Low: Minor damage</li>
                <li>Medium: Moderate damage</li>
                <li>High: Severe damage</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content area
    main_container = st.container()
    
    with main_container:
        # Prediction button
        if st.button('🔍 Analyze Damage', key='analyze_button'):
            try:
                # Prepare input (matches training data structure)
                input_data = {
                    'caliber': [float(caliber)],
                    'velocity': [float(velocity)],
                    'mass': [float(mass)],
                    'rifling_twist': [float(rifling_twist)],
                    'distance': [int(distance)],
                    'bullet_type': [bullet_type],
                    'gun_type': [gun_type],
                    'target_material': [target_material]
                }
            
                input_df = pd.DataFrame(input_data)
                
                # Single transform/predict call
                processed_input = preprocessor.transform(input_df)
                selected_input = selector.transform(processed_input)
                prediction = model.predict(selected_input)
                probabilities = model.predict_proba(selected_input)
                
                # ===== Results =====
                st.markdown("---")
                st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
                
                # Create a container for the main prediction
                with st.container():
                    # Determine color based on damage level
                    color_map = {
                        'Negative': '#4CAF50',  # Green
                        'Low': '#FFC107',       # Amber
                        'Medium': '#FF9800',    # Orange
                        'High': '#F44336'       # Red
                    }
                    color = color_map.get(class_names[prediction[0]], '#1E88E5')
                    
                    # Display prediction with custom styling
                    st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 1.5rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;">
                        <h2 style="margin: 0;">Predicted Damage Level</h2>
                        <h1 style="margin: 0.5rem 0;">{class_names[prediction[0]]}</h1>
                        <p style="margin: 0;">Confidence: {probabilities[0][prediction[0]]:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Confidence Analysis", "Feature Importance", "Input Summary", "Technical Details"])
                
                with tab1:
                    st.markdown('<h3 class="sub-header">Confidence Analysis</h3>', unsafe_allow_html=True)
                    
                    # Create a container for visualizations
                    viz_container = st.container()
                    
                    with viz_container:
                        # Create a more attractive probability visualization using Plotly
                        prob_df = pd.DataFrame({
                            'Damage Level': class_names,
                            'Confidence': probabilities[0]
                        })
                        
                        # Ensure data types are compatible with Streamlit
                        prob_df['Damage Level'] = prob_df['Damage Level'].astype(str)
                        prob_df['Confidence'] = prob_df['Confidence'].astype(float)
                        
                        fig = px.bar(
                            prob_df, 
                            x='Damage Level', 
                            y='Confidence',
                            color='Damage Level',
                            color_discrete_map=color_map,
                            title='Prediction Confidence by Damage Level',
                            labels={'Confidence': 'Confidence (%)', 'Damage Level': 'Damage Level'},
                            text=prob_df['Confidence'].apply(lambda x: f'{x:.2%}')
                        )
                        
                        fig.update_layout(
                            xaxis_title="Damage Level",
                            yaxis_title="Confidence",
                            yaxis_range=[0, 1],
                            yaxis_tickformat='.0%',
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(
                                size=14,
                                color='#E0E0E0'
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a gauge chart for the predicted class
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=float(probabilities[0][prediction[0]]) * 100,
                            title={'text': f"Confidence in {class_names[prediction[0]]} Prediction", 'font': {'color': '#E0E0E0'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickfont': {'color': '#E0E0E0'}},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 25], 'color': 'rgba(76, 175, 80, 0.2)'},
                                    {'range': [25, 50], 'color': 'rgba(255, 193, 7, 0.2)'},
                                    {'range': [50, 75], 'color': 'rgba(255, 152, 0, 0.2)'},
                                    {'range': [75, 100], 'color': 'rgba(244, 67, 54, 0.2)'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 80
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#E0E0E0'}
                        )
                        
                        st.plotly_chart(fig_gauge, use_container_width=True)
                
                with tab2:
                    st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
                    
                    # Show feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        with st.container():
                            importances = model.feature_importances_
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=True)
                            
                            # Ensure data types are compatible with Streamlit
                            importance_df['Feature'] = importance_df['Feature'].astype(str)
                            importance_df['Importance'] = importance_df['Importance'].astype(float)
                            
                            # Create a horizontal bar chart with Plotly
                            fig = px.bar(
                                importance_df, 
                                y='Feature', 
                                x='Importance',
                                orientation='h',
                                title='Feature Importance',
                                color='Importance',
                                color_continuous_scale='Blues'
                            )
                            
                            fig.update_layout(
                                xaxis_title="Importance",
                                yaxis_title="Feature",
                                height=400,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(
                                    size=12,
                                    color='#E0E0E0'
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add a summary of top features
                            st.markdown("### Top 5 Most Important Features")
                            top_features = importance_df.tail(5)
                            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                                st.markdown(f"**{i}.** {row['Feature']}: {row['Importance']:.4f}")
                    else:
                        st.info("Feature importance information is not available for this model.")
                
                with tab3:
                    st.markdown('<h3 class="sub-header">Input Summary</h3>', unsafe_allow_html=True)
                    
                    with st.container():
                        # Create a summary of the input parameters
                        st.markdown("### Firearm Parameters")
                        
                        # Create a two-column layout for the summary
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Physical Characteristics")
                            st.markdown(f"- **Caliber:** {caliber} mm")
                            st.markdown(f"- **Mass:** {mass} g")
                            st.markdown(f"- **Rifling Twist Rate:** {rifling_twist}")
                        
                        with col2:
                            st.markdown("#### Performance Characteristics")
                            st.markdown(f"- **Velocity:** {velocity} m/s")
                            st.markdown(f"- **Distance:** {distance} m")
                        
                        st.markdown("#### Categorical Parameters")
                        st.markdown(f"- **Bullet Type:** {bullet_type}")
                        st.markdown(f"- **Gun Type:** {gun_type}")
                        st.markdown(f"- **Target Material:** {target_material}")
                        
                        # Add a visualization of the input parameters
                        st.markdown("### Parameter Visualization")
                        
                        # Create a radar chart for numeric parameters
                        numeric_params = {
                            'Caliber': float(caliber),
                            'Velocity': float(velocity),
                            'Mass': float(mass),
                            'Rifling Twist': float(rifling_twist),
                            'Distance': float(distance)
                        }
                        
                        # Normalize values for radar chart
                        max_values = {
                            'Caliber': 15.0,
                            'Velocity': 1000.0,
                            'Mass': 20.0,
                            'Rifling Twist': 14.0,
                            'Distance': 100.0
                        }
                        
                        normalized_params = {k: v/max_values[k] for k, v in numeric_params.items()}
                        
                        # Create radar chart with Plotly
                        categories = list(normalized_params.keys())
                        values = list(normalized_params.values())
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Parameters'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1],
                                    tickfont={'color': '#E0E0E0'}
                                ),
                                angularaxis=dict(
                                    tickfont={'color': '#E0E0E0'}
                                )
                            ),
                            showlegend=False,
                            title={
                                'text': "Parameter Normalized Values",
                                'font': {'color': '#E0E0E0'}
                            },
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.markdown('<h3 class="sub-header">Technical Details</h3>', unsafe_allow_html=True)
                    
                    with st.container():
                        # Show model information first
                        st.markdown("""
                        <div class="info-box">
                        <h4 style="color: #E0E0E0;">Model Configuration</h4>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"- **Model Type:** {type(model).__name__}")
                        if hasattr(model, 'n_estimators'):
                            st.markdown(f"- **Number of Estimators:** {model.n_estimators}")
                        if hasattr(model, 'max_depth'):
                            st.markdown(f"- **Maximum Depth:** {model.max_depth}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Merge preprocessor features and selector support
                        st.markdown("""
                        <div class="info-box">
                        <h4 style="color: #E0E0E0;">Feature Analysis</h4>
                        """, unsafe_allow_html=True)
                        
                        # Create a DataFrame to show feature information
                        feature_info = pd.DataFrame({
                            'Feature Name': preprocessor.get_feature_names_out(),
                            'Selected': ['Yes' if s else 'No' for s in selector.support_],
                        })
                        
                        # Style the DataFrame for dark theme
                        def highlight_selected(val):
                            if val == 'Yes':
                                return 'background-color: rgba(76, 175, 80, 0.2); color: #4CAF50'
                            return 'background-color: rgba(244, 67, 54, 0.2); color: #F44336'
                        
                        styled_df = feature_info.style\
                            .applymap(highlight_selected, subset=['Selected'])\
                            .set_properties(**{
                                'background-color': 'rgba(30, 34, 45, 0.9)',
                                'color': '#E0E0E0',
                                'border': '1px solid rgba(255, 255, 255, 0.1)',
                                'padding': '0.5rem'
                            })
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Add feature statistics
                        total_features = len(feature_info)
                        selected_features = sum(selector.support_)
                        
                        st.markdown(f"""
                        #### Feature Statistics
                        - **Total Features:** {total_features}
                        - **Selected Features:** {selected_features}
                        - **Selection Ratio:** {selected_features/total_features:.1%}
                        """)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Add model limitations
                        st.markdown("""
                        <div class="info-box">
                        <h4 style="color: #E0E0E0;">Model Limitations</h4>
                        <p style="color: #E0E0E0;">This model is trained on historical ballistic data and may not account for all possible scenarios. 
                        The predictions should be used as a guide rather than definitive conclusions.</p>
                        
                        <h4 style="color: #E0E0E0;">Important Notes</h4>
                        <ul style="color: #E0E0E0;">
                            <li>The model's accuracy depends on the quality and coverage of the training data</li>
                            <li>Extreme or unusual combinations of parameters may lead to less reliable predictions</li>
                            <li>Regular model updates and validation are recommended for optimal performance</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add debug information in an expander
                        with st.expander("Debug Information"):
                            st.markdown("""
                            <div style='background-color: rgba(30, 34, 45, 0.9); padding: 1rem; border-radius: 0.5rem;'>
                            """, unsafe_allow_html=True)
                            st.write("Input Data Types:", str(input_df.dtypes))
                            st.write("Input Features:", list(input_df.columns))
                            st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"""
                Prediction Failed: {str(e)}
                Possible Issues:
                1. Feature mismatch (expected {len(feature_names)} features)
                2. Missing categories in input
                3. Model/preprocessor version mismatch
                """)

                with st.expander("Debug Details"):
                    st.write("Expected features:", preprocessor.get_feature_names_out())
                    st.write("Input features:", list(input_df.columns))
                    st.write("Input data types:", str(input_df.dtypes))
                    
                    # Check categories
                    try:
                        st.write("Preprocessor categories:")
                        st.write("Bullet types:", preprocessor.named_transformers_['cat']
                                .named_steps['onehot'].categories_[0])
                        st.write("Gun types:", preprocessor.named_transformers_['cat']
                                .named_steps['onehot'].categories_[1])
                    except:
                        pass

# Main function to run the entire pipeline
def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Initial data exploration
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset information:")
    print(df.info())
    
    print("\nDescriptive statistics:")
    print(df.describe())
    
    print("\nDamage level distribution:")
    print(df['damage_level'].describe())
    
    # Preprocess data
    df = preprocess_data(df)


    # Feature engineering
    X_processed, y_encoded, preprocessor, feature_names, class_names, y_cat = feature_engineering(df)
    
    # Feature selection
    selector = select_features(X_processed, y_encoded, feature_names)
    X_selected = selector.transform(X_processed)
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selector.support_[i]]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Train and evaluate models
    best_model, best_model_name, results_df = train_models(X_train, X_test, y_train, y_test, class_names)
    # Hyperparameter tuning for the best model
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        tuned_model = tune_hyperparameters(best_model, param_grid, X_train, y_train)
    elif best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        tuned_model = tune_hyperparameters(best_model, param_grid, X_train, y_train)

    else:
        tuned_model = best_model
    
    # Evaluate tuned model
    y_pred_tuned = tuned_model.predict(X_test)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    print(f"\nTuned {best_model_name} Accuracy: {accuracy_tuned:.2f}")
    
    # Plot learning curves
    plot_learning_curve(tuned_model, X_selected, y_encoded, f'Learning Curves ({best_model_name})')
    
    # Plot validation curves if applicable
    if best_model_name == 'Random Forest':
        plot_validation_curve(
            RandomForestClassifier(random_state=42),
            X_selected, y_encoded,
            param_name='n_estimators',
            param_range=[50, 100, 150, 200],
            title='Validation Curve for Random Forest (n_estimators)'
        )
    
    # Plot feature importance
    plot_feature_importance(tuned_model, selected_feature_names, f'{best_model_name} Feature Importance')
    
    # Save the best model and preprocessing objects
    joblib.dump(tuned_model, 'best_model.joblib')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    joblib.dump(selector, 'feature_selector.joblib')
    print("\nSaved best model, preprocessor, and feature selector to disk.")
    
    # Create Streamlit app
    


if __name__ == "__main__":
    try:
        # Load models
        tuned_model = joblib.load('best_model.joblib')
        preprocessor = joblib.load('preprocessor.joblib') 
        selector = joblib.load('feature_selector.joblib')
        
        # Get features
        class_names = ['Negative', 'Low', 'Medium', 'High']
        
        # Get all feature names from preprocessor
        all_feature_names = preprocessor.get_feature_names_out()
        
        # Get selected feature names based on selector support
        selected_feature_names = [f for f, s in zip(all_feature_names, selector.support_) if s]
        
        print("Loaded features:", selected_feature_names)
        print("Total features:", len(all_feature_names))
        print("Selected features:", len(selected_feature_names))
            
        create_streamlit_app(tuned_model, preprocessor, class_names, selected_feature_names)
        
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.error("Debug information:")
        st.write("Exception details:", str(e))
        if 'preprocessor' in locals():
            st.write("Preprocessor feature names:", preprocessor.get_feature_names_out())
        if 'selector' in locals():
            st.write("Selector support:", selector.support_)

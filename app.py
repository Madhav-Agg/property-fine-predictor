"""
Property Maintenance Fines Prediction - Streamlit Application

A web application for predicting blight ticket compliance in Detroit.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
import sys

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.app_functions import get_app_instance, create_sample_input, validate_input_data

# Configure Streamlit page
st.set_page_config(
    page_title="Property Maintenance Fines Prediction",
    page_icon=":building:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'app' not in st.session_state:
        st.session_state.app = get_app_instance()
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False


def display_header():
    """Display application header."""
    st.markdown('<h1 class="main-header">:building: Property Maintenance Fines Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    **Predict blight ticket compliance in Detroit using machine learning.**
    
    This application helps predict whether property maintenance fines will be paid on time,
    assisting the City of Detroit in optimizing enforcement strategies.
    """)


def display_sidebar():
    """Display sidebar with navigation and options."""
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Data Overview", "Model Training", "Single Prediction", "Batch Analysis", "Model Comparison"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Application Status**")
    
    if st.session_state.data_loaded:
        st.sidebar.success("Data loaded successfully")
        app = st.session_state.app
        if app.models:
            st.sidebar.info(f"Models available: {', '.join(app.models.keys())}")
        else:
            st.sidebar.warning("No models trained yet")
    else:
        st.sidebar.error("Data not loaded")
    
    return page


def initialize_data_page():
    """Display data initialization page."""
    st.header("Data Initialization")
    
    if not st.session_state.data_loaded:
        # Check if data files exist
        from src.utils import RAW_DATA_DIR
        data_files = ['train.csv', 'test.csv', 'addresses.csv', 'latlons.csv']
        missing_files = []
        
        for file in data_files:
            if not (RAW_DATA_DIR / file).exists():
                missing_files.append(file)
        
        if missing_files:
            st.error("Missing data files!")
            st.markdown("### Required Data Files")
            st.write("The following files are required in the `data/raw/` directory:")
            for file in data_files:
                status = "Missing" if file in missing_files else "Found"
                color = "red" if file in missing_files else "green"
                st.markdown(f"- <span style='color:{color}'>**{file}**</span>: {status}", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### How to Fix This")
            st.markdown("""
            1. **For local development**: Download the required CSV files and place them in the `data/raw/` directory
            2. **For Streamlit Cloud**: The data files should be included in the repository
            3. **Sample data**: If you're testing, you can create sample CSV files with the correct structure
            """)
            
            # Show expected structure
            with st.expander("Expected Data Structure"):
                st.markdown("""
                **train.csv** should contain:
                - ticket_id, compliance (target), and other features
                
                **test.csv** should contain:
                - ticket_id and the same features as train.csv (except compliance)
                
                **addresses.csv** should contain:
                - ticket_id, address
                
                **latlons.csv** should contain:
                - address, lat, lon
                """)
            return
        
        st.info("Click the button below to load and initialize the data. This may take a few minutes...")
        
        if st.button("Load Data", type="primary"):
            with st.spinner("Loading and preprocessing data..."):
                try:
                    app = st.session_state.app
                    result = app.initialize_data()
                    
                    if result['status'] == 'success':
                        st.session_state.data_loaded = True
                        st.success("Data loaded successfully!")
                        
                        # Display data summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Training Samples", result['train_samples'])
                        with col2:
                            st.metric("Test Samples", result['test_samples'])
                        with col3:
                            st.metric("Features", result['features'])
                        with col4:
                            st.metric("Models Loaded", len(result['models_loaded']))
                        
                        st.markdown("---")
                        st.subheader("Data Summary")
                        summary = app.get_data_summary()
                        if summary['status'] == 'success':
                            train_data = summary['training_data']
                            test_data = summary['test_data']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Training Data**")
                                st.write(f"- Samples: {train_data['samples']:,}")
                                st.write(f"- Features: {train_data['features']}")
                                st.write(f"- Compliance Rate: {train_data['compliance_rate']:.1%}")
                                st.write(f"- Compliant: {train_data['compliant_count']:,}")
                                st.write(f"- Non-compliant: {train_data['non_compliant_count']:,}")
                            
                            with col2:
                                st.markdown("**Test Data**")
                                st.write(f"- Samples: {test_data['samples']:,}")
                                st.write(f"- Features: {test_data['features']}")
                    else:
                        st.error(f"Failed to load data: {result['message']}")
                        
                        # Provide specific guidance based on error type
                        if "not found" in result['message'].lower():
                            st.markdown("---")
                            st.markdown("### Troubleshooting")
                            st.markdown("""
                            **Data files not found error:**
                            1. Ensure all CSV files are in the `data/raw/` directory
                            2. Check that file names match exactly: `train.csv`, `test.csv`, `addresses.csv`, `latlons.csv`
                            3. Verify the files are not empty
                            """)
                        
                except Exception as e:
                    st.error(f"Unexpected error during data loading: {str(e)}")
                    st.markdown("---")
                    st.markdown("### Troubleshooting")
                    st.markdown("""
                    **Unexpected error:**
                    1. Check the application logs for detailed error information
                    2. Ensure all dependencies are installed correctly
                    3. Try refreshing the page and loading data again
                    """)
    else:
        st.success("Data is already loaded!")
        
        app = st.session_state.app
        summary = app.get_data_summary()
        
        if summary['status'] == 'success':
            train_data = summary['training_data']
            test_data = summary['test_data']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Samples", f"{train_data['samples']:,}")
            with col2:
                st.metric("Test Samples", f"{test_data['samples']:,}")
            with col3:
                st.metric("Features", train_data['features'])
            with col4:
                st.metric("Compliance Rate", f"{train_data['compliance_rate']:.1%}")


def data_overview_page():
    """Display data overview page."""
    st.header("Data Overview")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first from the Home page.")
        return
    
    app = st.session_state.app
    summary = app.get_data_summary()
    
    if summary['status'] == 'success':
        train_data = summary['training_data']
        test_data = summary['test_data']
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Compliance Distribution")
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Compliant', 'Non-compliant'],
                values=[train_data['compliant_count'], train_data['non_compliant_count']],
                hole=0.3,
                marker_colors=['#2ecc71', '#e74c3c']
            )])
            fig.update_layout(title="Training Data Compliance")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Dataset Comparison")
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(name='Training', x=['Samples', 'Features'], y=[train_data['samples'], train_data['features']]),
                go.Bar(name='Test', x=['Samples', 'Features'], y=[test_data['samples'], test_data['features']])
            ])
            fig.update_layout(title="Dataset Sizes", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics
        st.subheader("Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Data Statistics**")
            st.write(f"- Total samples: {train_data['samples']:,}")
            st.write(f"- Number of features: {train_data['features']}")
            st.write(f"- Compliance rate: {train_data['compliance_rate']:.2%}")
            st.write(f"- Compliant tickets: {train_data['compliant_count']:,}")
            st.write(f"- Non-compliant tickets: {train_data['non_compliant_count']:,}")
        
        with col2:
            st.markdown("**Test Data Statistics**")
            st.write(f"- Total samples: {test_data['samples']:,}")
            st.write(f"- Number of features: {test_data['features']}")
            st.write(f"- Available models: {len(app.models)}")
            
            if app.models:
                st.write("**Trained Models:**")
                for model_type in app.models.keys():
                    st.write(f"- {model_type.replace('_', ' ').title()}")


def model_training_page():
    """Display model training page."""
    st.header("Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first from the Home page.")
        return
    
    app = st.session_state.app
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type:",
            ["gradient_boosting", "logistic_regression"],
            help="Choose the machine learning algorithm to train"
        )
    
    with col2:
        hyperparameter_tuning = st.checkbox(
            "Hyperparameter Tuning",
            help="Enable hyperparameter tuning for better performance (takes longer)"
        )
    
    # Check if model already exists
    model_exists = model_type in app.models
    
    if model_exists:
        st.info(f"Model {model_type} is already trained. You can retrain it with different settings.")
    
    # Training button
    if st.button("Train Model", type="primary", disabled=not st.session_state.data_loaded):
        with st.spinner(f"Training {model_type.replace('_', ' ').title()} model..."):
            result = app.train_model(model_type, hyperparameter_tuning)
            
            if result['status'] == 'success':
                st.success(result['message'])
                
                # Display training metrics
                metrics = result['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if 'cv_mean' in metrics:
                        st.metric("CV AUC", f"{metrics['cv_mean']:.4f}")
                with col2:
                    if 'train_auc' in metrics:
                        st.metric("Train AUC", f"{metrics['train_auc']:.4f}")
                with col3:
                    if 'train_accuracy' in metrics:
                        st.metric("Train Accuracy", f"{metrics['train_accuracy']:.4f}")
                with col4:
                    if 'best_score' in metrics:
                        st.metric("Best Score", f"{metrics['best_score']:.4f}")
                
                # Display feature importance if available
                if 'top_features' in metrics:
                    st.subheader("Top Features")
                    features_df = pd.DataFrame(
                        list(metrics['top_features'].items()),
                        columns=['Feature', 'Importance']
                    )
                    st.bar_chart(features_df.set_index('Feature'))
                
                # Display best parameters if hyperparameter tuning was used
                if hyperparameter_tuning and 'best_params' in metrics:
                    st.subheader("Best Hyperparameters")
                    for param, value in metrics['best_params'].items():
                        st.write(f"- {param}: {value}")
                
            else:
                st.error(result['message'])
    
    # Display model information
    if model_type in app.models:
        st.markdown("---")
        st.subheader("Model Information")
        
        model_info = app.get_model_info(model_type)
        
        if model_info['status'] == 'success':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", model_info['model_class'])
            with col2:
                st.metric("Features", model_info['feature_columns'])
            with col3:
                st.metric("Trained", "Yes" if model_info['is_trained'] else "No")
            
            if 'top_features' in model_info:
                st.subheader("Feature Importance")
                features_df = pd.DataFrame(
                    list(model_info['top_features'].items()),
                    columns=['Feature', 'Importance']
                )
                st.dataframe(features_df, use_container_width=True)


def single_prediction_page():
    """Display single prediction page."""
    st.header("Single Ticket Prediction")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first from the Home page.")
        return
    
    app = st.session_state.app
    
    # Check if models are available, if not try to train them
    if not app.models:
        st.warning("No trained models available. Training default models...")
        
        with st.spinner("Training default models for prediction..."):
            # Try to train both models
            models_trained = []
            for model_type in ['logistic_regression', 'gradient_boosting']:
                try:
                    app._train_fallback_model(model_type)
                    models_trained.append(model_type)
                except Exception as e:
                    st.error(f"Failed to train {model_type}: {str(e)}")
        
        if models_trained:
            st.success(f"Successfully trained: {', '.join(models_trained)}")
        else:
            st.error("Could not train any models. Please go to Model Training page to train manually.")
            return
    
    # Model selection
    model_type = st.selectbox(
        "Select Model:",
        list(app.models.keys()),
        help="Choose the trained model to use for prediction"
    )
    
    st.markdown("---")
    st.subheader("Input Ticket Information")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic information
        st.markdown("**Basic Information**")
        ticket_id = st.number_input("Ticket ID", value=12345, min_value=1)
        agency_name = st.selectbox(
            "Agency Name",
            ["Department of Health", "Department of Buildings", "Police Department"]
        )
        violation_code = st.text_input("Violation Code", value="CODE_1")
        violation_description = st.selectbox(
            "Violation Description",
            ["Failure to maintain exterior", "Improper waste disposal", "Safety violations", "Other"]
        )
        disposition = st.selectbox(
            "Disposition",
            ["Responsible", "Not Responsible", "Pending"]
        )
    
    with col2:
        # Financial information
        st.markdown("**Financial Information**")
        fine_amount = st.number_input("Fine Amount ($)", value=250.0, min_value=0.0, step=10.0)
        admin_fee = st.number_input("Admin Fee ($)", value=20.0, min_value=0.0, step=5.0)
        state_fee = st.number_input("State Fee ($)", value=10.0, min_value=0.0, step=5.0)
        late_fee = st.number_input("Late Fee ($)", value=25.0, min_value=0.0, step=5.0)
        discount_amount = st.number_input("Discount Amount ($)", value=0.0, min_value=0.0, step=5.0)
        clean_up_cost = st.number_input("Clean-up Cost ($)", value=0.0, min_value=0.0, step=10.0)
    
    # Location and timing
    st.markdown("**Location and Timing**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lat = st.number_input("Latitude", value=42.3314, min_value=-90.0, max_value=90.0)
    with col2:
        lon = st.number_input("Longitude", value=-83.0458, min_value=-180.0, max_value=180.0)
    with col3:
        hearing_issued_date_diff = st.number_input(
            "Days Between Hearing and Issue", 
            value=30, 
            min_value=0, 
            help="Number of days between hearing date and ticket issue date"
        )
    
    # Calculate judgment amount
    judgment_amount = fine_amount + admin_fee + state_fee + late_fee - discount_amount + clean_up_cost
    
    # Prediction button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predict Compliance", type="primary"):
            # Prepare input data
            input_data = {
                'ticket_id': ticket_id,
                'agency_name': agency_name,
                'violation_code': violation_code,
                'violation_description': violation_description,
                'disposition': disposition,
                'fine_amount': fine_amount,
                'admin_fee': admin_fee,
                'state_fee': state_fee,
                'late_fee': late_fee,
                'discount_amount': discount_amount,
                'clean_up_cost': clean_up_cost,
                'judgment_amount': judgment_amount,
                'lat': lat,
                'lon': lon,
                'hearing_issued_date_diff': hearing_issued_date_diff
            }
            
            # Validate input
            is_valid, errors = validate_input_data(input_data)
            
            if not is_valid:
                st.error("Please fix the following errors:")
                for error in errors:
                    st.write(f"- {error}")
            else:
                # Make prediction
                with st.spinner("Making prediction..."):
                    result = app.predict_single(model_type, input_data)
                    
                    if result['status'] == 'success':
                        # Display results
                        st.markdown("---")
                        st.subheader("Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Probability gauge
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = result['probability'] * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Compliance Probability (%)"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "#1f77b4"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"},
                                        {'range': [80, 100], 'color': "lightgreen"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Prediction result
                            prediction_color = "green" if result['prediction'] == 'compliant' else "red"
                            st.markdown(f"<h3 style='color: {prediction_color}; text-align: center;'>"
                                      f"Prediction: {result['prediction'].upper()}</h3>", 
                                      unsafe_allow_html=True)
                            
                            st.markdown(f"<h4 style='text-align: center;'>"
                                      f"Probability: {result['probability']:.1%}</h4>", 
                                      unsafe_allow_html=True)
                            
                            confidence_color = {
                                'high': '#2ecc71',
                                'medium': '#f39c12', 
                                'low': '#e74c3c'
                            }.get(result['confidence'], '#95a5a6')
                            
                            st.markdown(f"<p style='text-align: center; color: {confidence_color};'>"
                                      f"Confidence: {result['confidence'].upper()}</p>", 
                                      unsafe_allow_html=True)
                        
                        with col3:
                            # Summary metrics
                            st.markdown("**Summary**")
                            st.write(f"- **Model**: {model_type.replace('_', ' ').title()}")
                            st.write(f"- **Prediction**: {result['prediction']}")
                            st.write(f"- **Probability**: {result['probability']:.1%}")
                            st.write(f"- **Confidence**: {result['confidence']}")
                            st.write(f"- **Judgment Amount**: ${judgment_amount:.2f}")
                    
                    else:
                        st.error(result['message'])
    
    # Load sample data button
    st.markdown("---")
    if st.button("Load Sample Data"):
        sample_data = create_sample_input()
        st.info("Sample data loaded! You can modify the values above and then click 'Predict Compliance'.")
        # Note: In a real implementation, we'd use st.session_state to populate the fields


def batch_analysis_page():
    """Display batch analysis page."""
    st.header("Batch Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first from the Home page.")
        return
    
    app = st.session_state.app
    
    if not app.models:
        st.warning("No trained models available. Please train a model first.")
        return
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Model:",
            list(app.models.keys()),
            help="Choose the trained model to use for batch analysis"
        )
    
    with col2:
        sample_size = st.number_input(
            "Sample Size (0 for all)",
            value=1000,
            min_value=0,
            max_value=len(app.test_data),
            help="Number of test samples to analyze (0 = use all)"
        )
    
    # Analysis button
    if st.button("Run Batch Analysis", type="primary"):
        with st.spinner("Running batch analysis..."):
            result = app.predict_batch(model_type, sample_size if sample_size > 0 else None)
            
            if result['status'] == 'success':
                summary = result['summary']
                
                # Display summary metrics
                st.subheader("Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Predictions", f"{summary['total_predictions']:,}")
                with col2:
                    st.metric("Mean Probability", f"{summary['mean_probability']:.1%}")
                with col3:
                    st.metric("Compliant", f"{summary['compliant_count']:,}")
                with col4:
                    st.metric("Non-compliant", f"{summary['non_compliant_count']:,}")
                
                # Create visualizations
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability distribution
                    st.subheader("Probability Distribution")
                    
                    # Create histogram data
                    all_predictions = list(result['all_predictions'].values())
                    
                    fig = go.Figure(data=[go.Histogram(
                        x=all_predictions,
                        nbinsx=20,
                        marker_color='#1f77b4',
                        opacity=0.7
                    )])
                    fig.update_layout(
                        title="Distribution of Compliance Probabilities",
                        xaxis_title="Probability",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Compliance breakdown
                    st.subheader("Compliance Breakdown")
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Compliant', 'Non-compliant'],
                        values=[summary['compliant_count'], summary['non_compliant_count']],
                        hole=0.3,
                        marker_colors=['#2ecc71', '#e74c3c']
                    )])
                    fig.update_layout(title="Predicted Compliance")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sample predictions table
                st.markdown("---")
                st.subheader("Sample Predictions")
                
                sample_predictions = result['sample_predictions']
                predictions_df = pd.DataFrame([
                    {
                        'Ticket ID': ticket_id,
                        'Probability': f"{prob:.1%}",
                        'Prediction': 'Compliant' if prob > 0.5 else 'Non-compliant'
                    }
                    for ticket_id, prob in list(sample_predictions.items())[:10]
                ])
                
                st.dataframe(predictions_df, use_container_width=True)
                
            else:
                st.error(result['message'])


def model_comparison_page():
    """Display model comparison page."""
    st.header("Model Comparison")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first from the Home page.")
        return
    
    app = st.session_state.app
    
    if len(app.models) < 2:
        st.warning("At least two trained models are needed for comparison. Please train more models.")
        return
    
    # Comparison button
    if st.button("Compare Models", type="primary"):
        with st.spinner("Comparing models..."):
            result = app.compare_models()
            
            if result['status'] == 'success':
                comparison = result['comparison']
                
                # Create comparison table
                st.subheader("Model Performance Comparison")
                
                comparison_df = pd.DataFrame(comparison).T
                comparison_df = comparison_df[['cv_auc', 'train_auc', 'train_accuracy', 'n_samples', 'n_features']]
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Create visualization
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # AUC comparison
                    st.subheader("AUC Score Comparison")
                    
                    models = list(comparison.keys())
                    cv_auc_scores = [float(comparison[model]['cv_auc'].split(' ')[0]) for model in models]
                    train_auc_scores = [float(comparison[model]['train_auc']) for model in models]
                    
                    fig = go.Figure(data=[
                        go.Bar(name='CV AUC', x=models, y=cv_auc_scores),
                        go.Bar(name='Train AUC', x=models, y=train_auc_scores)
                    ])
                    fig.update_layout(
                        title="AUC Score Comparison",
                        barmode='group',
                        yaxis_title="AUC Score"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Feature importance comparison (if available)
                    st.subheader("Top Features by Model")
                    
                    for model_type in models:
                        if 'top_features' in comparison[model_type]:
                            st.markdown(f"**{model_type.replace('_', ' ').title()}**")
                            features = comparison[model_type]['top_features']
                            for feature, importance in list(features.items())[:3]:
                                st.write(f"- {feature}: {importance:.3f}")
                            st.write("")
                
            else:
                st.error(result['message'])


def home_page():
    """Display home page."""
    st.header("Welcome to Property Maintenance Fines Prediction")
    
    # Initialize data if not done
    if not st.session_state.data_loaded:
        st.info("Click 'Initialize Data' below to get started, or navigate to other pages using the sidebar.")
        
        if st.button("Initialize Data", type="primary"):
            with st.spinner("Loading and preprocessing data..."):
                app = st.session_state.app
                result = app.initialize_data()
                
                if result['status'] == 'success':
                    st.session_state.data_loaded = True
                    st.success("Data loaded successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to load data: {result['message']}")
    
    if st.session_state.data_loaded:
        app = st.session_state.app
        
        # Display overview
        st.markdown("---")
        st.subheader("Application Overview")
        
        summary = app.get_data_summary()
        
        if summary['status'] == 'success':
            train_data = summary['training_data']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Samples", f"{train_data['samples']:,}")
            with col2:
                st.metric("Features", train_data['features'])
            with col3:
                st.metric("Compliance Rate", f"{train_data['compliance_rate']:.1%}")
            with col4:
                st.metric("Models", len(app.models))
        
        # Quick actions
        st.markdown("---")
        st.subheader("Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Train Model", type="secondary"):
                st.info("Navigate to 'Model Training' to train a model.")
        
        with col2:
            if st.button("Make Prediction", type="secondary"):
                st.info("Navigate to 'Single Prediction' to make predictions.")
        
        with col3:
            if st.button("View Data", type="secondary"):
                st.info("Navigate to 'Data Overview' to view data statistics.")
        
        # Model status
        if app.models:
            st.markdown("---")
            st.subheader("Available Models")
            
            for model_type in app.models.keys():
                model_info = app.get_model_info(model_type)
                if model_info['status'] == 'success':
                    with st.expander(f"{model_type.replace('_', ' ').title()}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"- **Type**: {model_info['model_class']}")
                            st.write(f"- **Features**: {model_info['feature_columns']}")
                        with col2:
                            st.write(f"- **Trained**: {'Yes' if model_info['is_trained'] else 'No'}")
                        
                        if 'top_features' in model_info:
                            st.write("**Top Features:**")
                            for feature, importance in list(model_info['top_features'].items())[:5]:
                                st.write(f"- {feature}: {importance:.3f}")


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar and get selected page
    page = display_sidebar()
    
    # Route to appropriate page
    if page == "Home":
        home_page()
    elif page == "Data Overview":
        data_overview_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Model Comparison":
        model_comparison_page()
    else:
        st.error("Page not found")


if __name__ == "__main__":
    main()

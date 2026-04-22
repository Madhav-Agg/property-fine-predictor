# Property Maintenance Fines Prediction - Streamlit Application

A modern, interactive web application for predicting blight ticket compliance in Detroit using machine learning. Built with Streamlit for an intuitive user interface.

## Authors

Created by: Diya, Madhav, and Sushant

## Project Overview

This application leverages machine learning to predict whether blight tickets issued by the City of Detroit will be paid on time. By analyzing historical ticket data, the system helps optimize enforcement strategies and reduce costs associated with unpaid fines.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Create project directory:**
   ```bash
   mkdir property-maintenance-fines
   cd property-maintenance-fines
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and place data files:**
   - Obtain the required CSV files from the official Detroit Open Data Portal:
     - `train.csv` - Training data (2004-2011 tickets)
     - `test.csv` - Test data (2012-2016 tickets)
     - `addresses.csv` - Address mapping
     - `latlons.csv` - Latitude/longitude coordinates
   - Place them in the `data/raw/` directory

### Running the Application

Start the Streamlit app with:

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## Application Features

### Multi-Page Interface
- **Home**: Data initialization and dashboard
- **Data Overview**: Statistics and visualizations
- **Model Training**: Train ML models with hyperparameter tuning
- **Single Prediction**: Interactive form for individual predictions
- **Batch Analysis**: Process multiple predictions at once
- **Model Comparison**: Compare different model performances

### Interactive Elements
- Real-time data loading and processing
- Interactive input forms with validation
- Visual charts using Plotly
- Progress indicators and spinners
- Color-coded prediction results
- Responsive design for all screen sizes

### Advanced Functionality
- Model persistence and loading
- Feature importance visualization
- Batch prediction with statistical summaries
- Cross-validation metrics
- Hyperparameter optimization
- Error handling and logging

## Using the Application

### Step 1: Initialize Data
1. Navigate to the **Home** page
2. Click **"Initialize Data"** to load and preprocess the dataset
3. Wait for the data loading to complete (may take a few minutes)

### Step 2: Train a Model
1. Go to **Model Training** page
2. Select your preferred algorithm (Gradient Boosting recommended)
3. Optionally enable hyperparameter tuning for better performance
4. Click **"Train Model"** and wait for training to complete

### Step 3: Make Predictions

#### For Single Tickets:
1. Navigate to **Single Prediction**
2. Fill in the ticket information using the interactive form
3. Click **"Predict Compliance"** to get instant results
4. View the probability and confidence level

#### For Batch Analysis:
1. Go to **Batch Analysis**
2. Select your trained model
3. Choose sample size (or use all data)
4. Click **"Run Batch Analysis"** for comprehensive results

### Step 4: Compare Models (Optional)
1. Train at least two different models
2. Navigate to **Model Comparison**
3. Click **"Compare Models"** to see performance differences
4. Analyze which model performs best for your needs

## Model Performance

Based on our analysis of Detroit blight ticket data:

| Model | CV AUC Score | Training AUC | Best For |
|-------|--------------|--------------|----------|
| Gradient Boosting | ~0.806 | ~0.85 | Highest accuracy |
| Logistic Regression | ~0.787 | ~0.82 | Fast training, interpretability |

## Input Fields Guide

### Required Fields for Prediction:

#### Basic Information
- **Agency Name**: Department issuing the ticket
- **Violation Code**: Code for the violation type
- **Violation Description**: Description of the violation
- **Disposition**: Case outcome (Responsible, Not Responsible, Pending)

#### Financial Information
- **Fine Amount**: Base fine amount ($)
- **Admin Fee**: Administrative fee ($)
- **State Fee**: State processing fee ($)
- **Late Fee**: Late payment penalty ($)
- **Discount Amount**: Any applicable discount ($)
- **Clean-up Cost**: Additional clean-up charges ($)

#### Location and Timing
- **Latitude**: Geographic latitude
- **Longitude**: Geographic longitude
- **Days Between Hearing and Issue**: Time difference in days

## Understanding Results

### Prediction Output
- **Probability**: Likelihood of compliance (0-100%)
- **Prediction**: Binary outcome (Compliant/Non-compliant)
- **Confidence**: Assessment of prediction reliability

### Color Coding
- **Green**: Compliant prediction / High confidence
- **Red**: Non-compliant prediction / Low confidence
- **Yellow**: Medium confidence

## Project Structure

```
property-maintenance-fines/
|
|-- app.py                          # Main Streamlit application
|-- src/
|   |-- app_functions.py            # Core application logic
|   |-- preprocessing/               # Data processing modules
|   |-- models/                     # ML model implementations
|   |-- utils/                       # Utilities and configuration
|
|-- data/
|   |-- raw/                        # Raw CSV files (add your data here)
|   |-- processed/                  # Processed data (auto-generated)
|
|-- models/                         # Trained models (auto-generated)
|-- logs/                           # Application logs (auto-generated)
|-- tests/                          # Unit tests
|-- requirements.txt                # Python dependencies
|-- .streamlit/                     # Streamlit configuration
|-- README.md                       # This file
```

## Troubleshooting

### Common Issues

1. **Data Loading Fails**:
   - Ensure all CSV files are in `data/raw/`
   - Check file names match exactly
   - Verify data files are not corrupted

2. **Model Training Errors**:
   - Make sure data is loaded first
   - Check for sufficient memory
   - Review log files in `logs/` directory

3. **Prediction Fails**:
   - Verify all required fields are filled
   - Check numeric values are valid
   - Ensure a model is trained first

4. **Application Won't Start**:
   - Verify all dependencies are installed
   - Check Python version compatibility
   - Ensure virtual environment is activated

## Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Configure environment variables if needed
4. Deploy!

### Custom Port
```bash
streamlit run app.py --server.port 8080
```

## Configuration

The application uses `.streamlit/config.toml` for customization:

- **Theme**: Colors and styling
- **Server**: Port and CORS settings
- **Logging**: Verbosity levels

Modify the file to customize the appearance and behavior.

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ app.py
```

### Linting
```bash
flake8 src/ app.py
```

## Security Considerations

- No sensitive data is stored permanently
- All processing happens in memory
- No external API calls are made
- Input validation prevents injection attacks

## Contributing

1. Fork repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Contributors

- Diya 
- Madhav
- Sushant 

## License

This project is provided for educational and research purposes.

---

**Need Help?** Check the troubleshooting section or open an issue on GitHub.

#!/usr/bin/env python3
"""
Test model loading and prediction functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_model_loading():
    print("=== TESTING MODEL LOADING ===")
    
    from src.app_functions import get_app_instance
    from src.utils import MODEL_DIR
    
    print(f"Model directory: {MODEL_DIR}")
    print("Model files in directory:")
    for file in MODEL_DIR.glob('*.pkl'):
        print(f"  - {file.name} ({file.stat().st_size} bytes)")
    
    # Test model loading
    app = get_app_instance()
    result = app.initialize_data()
    
    if result['status'] == 'success':
        print(f"Data initialized: {result['train_samples']} samples")
        print(f"Loaded models: {list(app.models.keys())}")
        
        # Test prediction with existing model
        if app.models:
            model_type = list(app.models.keys())[0]
            print(f"Testing prediction with {model_type} model...")
            
            sample_input = {
                'agency_name': 'Police',
                'violation_code': 'CODE_0',
                'violation_description': 'Improper waste disposal',
                'disposition': 'Pending',
                'fine_amount': 100.0,
                'admin_fee': 20.0,
                'state_fee': 10.0,
                'late_fee': 5.0,
                'discount_amount': 0.0,
                'clean_up_cost': 0.0,
                'lat': 42.3314,
                'lon': -83.0458,
                'hearing_issued_date_diff': 30
            }
            
            pred_result = app.predict_single(model_type, sample_input)
            print(f"Prediction result: {pred_result['status']}")
            if pred_result['status'] == 'success':
                print(f"Probability: {pred_result['probability']:.3f}")
                return True
            else:
                print(f"Error: {pred_result['message']}")
                return False
        else:
            print("No models loaded!")
            return False
    else:
        print(f"Data initialization failed: {result['message']}")
        return False

def test_model_training():
    print("\n=== TESTING MODEL TRAINING ===")
    
    from src.app_functions import get_app_instance
    
    app = get_app_instance()
    if not app.is_initialized:
        result = app.initialize_data()
        if result['status'] != 'success':
            print(f"Cannot test training - data initialization failed")
            return False
    
    # Test training a new model
    print("Training logistic regression model...")
    train_result = app.train_model('logistic_regression', hyperparameter_tuning=False)
    
    if train_result['status'] == 'success':
        print("Model training successful!")
        print(f"CV AUC: {train_result['metrics']['cv_mean']:.3f}")
        return True
    else:
        print(f"Model training failed: {train_result['message']}")
        return False

def main():
    success1 = test_model_loading()
    success2 = test_model_training()
    
    if success1 and success2:
        print("\n=== ALL MODEL TESTS PASSED ===")
        return True
    else:
        print("\n=== SOME MODEL TESTS FAILED ===")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

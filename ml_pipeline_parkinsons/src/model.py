import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_model(model_type='random_forest'):
    """
    Returns an instantiated sklearn classifier for Parkinson's detection.
    
    Args:
        model_type (str): 'random_forest' or 'gradient_boosting'
        
    Returns:
        Classifier instance
    """
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
    else:
        raise ValueError(f"Model type '{model_type}' is not supported. Use 'random_forest' or 'gradient_boosting'.")

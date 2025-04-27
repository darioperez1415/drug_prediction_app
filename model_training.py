import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def prepare_data(df_path):
    """Load and prepare the dataset"""
    df = pd.read_csv(df_path)
    
    # Define features and targets
    features = ['age', 'gender', 'education', 'country', 'ethnicity',
                'nscore', 'escore', 'oscore', 'ascore', 'cscore',
                'impuslive', 'ss']
    
    drug_columns = [col for col in df.columns if '_target' in col]
    
    # Create binary targets
    for drug in drug_columns:
        df[f"{drug}_binary"] = df[drug].apply(lambda x: 0 if x in ["CL0", "CL1"] else 1)
    
    return df, features, [f"{d}_binary" for d in drug_columns]

def train_models(df, features, drug_targets, min_samples=10):
    """Train models for all drugs meeting minimum sample size"""
    results = []
    
    for target in drug_targets:
        if df[target].sum() < min_samples:
            print(f"Skipping {target}: insufficient samples")
            continue
            
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Handle class imbalance
        minority_count = y_train.value_counts().min()
        if minority_count <= 5:
            pipeline = Pipeline([
                ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
            ])
        else:
            k_neighbors = max(1, min(3, minority_count - 1))
            pipeline = Pipeline([
                ('smote', SMOTE(k_neighbors=k_neighbors, random_state=42)),
                ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
            ])
        
        # Train model
        params = {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [5, 10, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__criterion': ['gini', 'entropy']
        }
        
        grid_search = GridSearchCV(pipeline, params, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Store results
        results.append({
            'Drug': target.replace('_binary', ''),
            'model': grid_search,
            'X_test': X_test,
            'y_test': y_test
        })
    
    return results

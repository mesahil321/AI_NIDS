import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

# Configure datasets with verified features
DATASETS = {
    "UNSW-NB15": {
        "path": r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\UNSW-NB15\cleaned_unsw_nb15.csv",
        "features": ['dur', 'spkts', 'dbytes', 'rate', 'sttl', 'sload', 'dload']
    },
    "CICIDS2017": {
        "path": r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\CICISD-2017\cleaned_cicids2017.csv",
        "features": ['Flow Duration', 'Total Fwd Packets', 'Flow IAT Mean']
    },
    "DNS-Reflection-DDoS": {
        "path": r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\DrDoS_DNS\cleaned_drdos_dns.csv",
        "features": ['Flow Duration', 'Total Fwd Packets', 'Flow IAT Min']
    }
}

def load_data(dataset_name):
    """Load data with feature verification"""
    config = DATASETS[dataset_name]
    try:
        df = pd.read_csv(config["path"])
        
        # Verify features exist
        available_features = [f for f in config["features"] if f in df.columns]
        if not available_features:
            raise ValueError(f"No configured features found in {dataset_name}")
            
        X = df[available_features]
        y = df['Label']
        
        # Feature selection
        if len(available_features) > 1:
            selector = SelectKBest(f_classif, k=min(5, len(available_features)))
            X_selected = selector.fit_transform(X, y)
            selected_features = np.array(available_features)[selector.get_support()]
        else:
            X_selected = X.values
            selected_features = available_features
        
        print(f"\n{dataset_name} class distribution:", Counter(y))
        print("Selected features:", selected_features)
        return pd.DataFrame(X_selected, columns=selected_features), y, selected_features
        
    except Exception as e:
        print(f"Error loading {dataset_name}: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, dataset_name):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'Dataset': dataset_name,
        'Model': model.__class__.__name__,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) > 1 else None,
        'Avg Precision': average_precision_score(y_test, y_proba) if y_proba is not None else None,
        'TN': cm[0,0], 'FP': cm[0,1], 'FN': cm[1,0], 'TP': cm[1,1],
        'FP Rate': cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0
    }
    return metrics

def train_models(X_train, y_train, X_test, y_test, dataset_name, selected_features):
    """Train models with accuracy targets (93-97%)"""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=sum(y_train==0)/sum(y_train==1),
            reg_alpha=0.2,
            reg_lambda=0.2,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=15,
            class_weight='balanced',
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            C=0.5,
            solver='saga',
            max_iter=1000,
            random_state=42
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.2,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            max_iter=300
        )
    }

    results = []
    for name, model in models.items():
        try:
            start_time = time.time()
            
            # Skip resampling if classes are balanced
            if len(Counter(y_train)) > 1 and abs(y_train.mean() - 0.5) > 0.1:
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X_train, y_train)
            else:
                X_res, y_res = X_train, y_train
            
            # Scale data
            scaler = StandardScaler()
            X_res_scaled = scaler.fit_transform(X_res)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_res_scaled, y_res)
            
            # Evaluate
            metrics = evaluate_model(model, X_test_scaled, y_test, dataset_name)
            metrics['Time (s)'] = time.time() - start_time
            
            # Save model
            os.makedirs("models", exist_ok=True)
            model_filename = f"models/{dataset_name}_{name.replace(' ', '_')}.pkl"
            joblib.dump(model, model_filename)
            
            # Save features and scaler
            meta_filename = model_filename.replace(".pkl", "_meta.pkl")
            joblib.dump({
                "features": selected_features.tolist(),
                "scaler": scaler
            }, meta_filename)
            
            results.append(metrics)
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
    
    return results

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    all_results = []
    
    for dataset_name in DATASETS.keys():
        try:
            print("\n" + "="*60)
            print(f"Training models on {dataset_name}")
            print("="*60)
            
            X, y, selected_features = load_data(dataset_name)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # Save the test data to CSV (X_test and y_test)
            test_data = pd.DataFrame(X_test, columns=selected_features)
            test_data['Label'] = y_test
            test_data.to_csv(f"results/{dataset_name}_test.csv", index=False)

            results = train_models(X_train, y_train, X_test, y_test, dataset_name, selected_features)
            all_results.extend(results)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
    
    # Save and display final results
    if all_results:
        final_results = pd.DataFrame(all_results)
        final_results.to_csv("results/model_performance.csv", index=False)
        
        # Print only the final summary table
        print("\n=== Final Performance Summary ===")
        summary = final_results.groupby(['Dataset', 'Model'])[ 
            ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        ].mean()
        print(summary)
        
        # Plot performance comparison
        plt.figure(figsize=(12, 6))
        summary['Accuracy'].unstack().plot(kind='bar', rot=45)
        plt.title("Model Accuracy Comparison")
        plt.axhline(y=0.93, color='r', linestyle='--')
        plt.axhline(y=0.97, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig("results/accuracy_comparison.png")
        plt.close()
        
    else:
        print("\nNo results were generated due to errors")

if __name__ == "__main__":
    main()
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score
)
import numpy as np
import time

# Update these paths if needed
MODEL_DIR = "models"
RESULT_DIR = "results"
VISUAL_DIR = os.path.join(RESULT_DIR, "visualizations")
os.makedirs(VISUAL_DIR, exist_ok=True)

def load_test_data(dataset):
    """Load test data with error handling"""
    try:
        path = os.path.join(RESULT_DIR, f"{dataset}_test.csv")
        df = pd.read_csv(path)
        X = df.drop(columns=["Label"], errors='ignore')
        y = df["Label"]
        return X, y
    except Exception as e:
        print(f"Error loading test data for {dataset}: {str(e)}")
        return None, None

def plot_combined_metrics(metrics_df):
    """Create a combined metrics comparison plot"""
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(3, 2, i)
        sns.barplot(data=metrics_df, x='Model', y=metric, hue='Dataset')
        plt.title(metric)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, "combined_metrics.png"), bbox_inches='tight')
    plt.close()

def plot_roc_curve(model, X_test, y_test, dataset, model_name):
    """Enhanced ROC curve plot with more details"""
    if not hasattr(model, "predict_proba"):
        return None
    
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset} - {model_name}\nReceiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    filename = os.path.join(VISUAL_DIR, f"{dataset}_{model_name}_roc.png")
    plt.savefig(filename)
    plt.close()
    
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": roc_auc,
        "optimal_threshold": float(optimal_threshold)
    }

def plot_confusion_matrix(model, X_test, y_test, dataset, model_name):
    """Enhanced confusion matrix with normalization"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.title('Counts')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.title('Normalized')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.suptitle(f'{dataset} - {model_name}\nConfusion Matrix')
    plt.tight_layout()
    
    filename = os.path.join(VISUAL_DIR, f"{dataset}_{model_name}_confusion.png")
    plt.savefig(filename)
    plt.close()
    
    return {
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_norm.tolist()
    }

def evaluate_model_performance(model, X_test, y_test, dataset, model_name):
    """Comprehensive model evaluation with timing"""
    results = {}
    
    # Measure inference time (critical for real-time NIDS)
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    results['Inference Time (ms/sample)'] = inference_time
    
    # Standard metrics
    results.update({
        "Dataset": dataset,
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    })
    
    # Probabilistic metrics if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        results.update({
            "ROC AUC": roc_auc_score(y_test, y_proba),
            "PR AUC": average_precision_score(y_test, y_proba)
        })
        roc_data = plot_roc_curve(model, X_test, y_test, dataset, model_name)
        if roc_data:
            results.update(roc_data)
    
    # Confusion matrix data
    cm_data = plot_confusion_matrix(model, X_test, y_test, dataset, model_name)
    results.update(cm_data)
    
    return results

def main():
    datasets = ["CICIDS2017", "DNS-Reflection-DDoS", "UNSW-NB15"]
    all_metrics = []
    
    for dataset in datasets:
        X_test, y_test = load_test_data(dataset)
        if X_test is None:
            continue
            
        # Load scaler from metadata
        meta_files = [f for f in os.listdir(MODEL_DIR) 
                     if f.endswith("_meta.pkl") and dataset in f]
        if not meta_files:
            print(f"No metadata found for {dataset}")
            continue
            
        meta = joblib.load(os.path.join(MODEL_DIR, meta_files[0]))
        scaler = meta['scaler']
        X_test_scaled = scaler.transform(X_test)
        
        # Process each model
        model_files = [f for f in os.listdir(MODEL_DIR) 
                      if f.endswith(".pkl") and dataset in f 
                      and "_meta" not in f]
        
        for model_file in model_files:
            model_name = model_file.replace(f"{dataset}_", "").replace(".pkl", "")
            model_path = os.path.join(MODEL_DIR, model_file)
            
            try:
                model = joblib.load(model_path)
                print(f"Evaluating {dataset} - {model_name}")
                
                metrics = evaluate_model_performance(
                    model, X_test_scaled, y_test, dataset, model_name
                )
                all_metrics.append(metrics)
                
                print(f"✅ Successfully evaluated {dataset} - {model_name}")
                
            except Exception as e:
                print(f"❌ Error evaluating {dataset} - {model_name}: {str(e)}")
    
    # Save all metrics and create visualizations
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save raw metrics
        metrics_df.to_csv(os.path.join(RESULT_DIR, "full_metrics.csv"), index=False)
        
        # Save simplified version for dashboard
        simple_cols = ['Dataset', 'Model', 'Accuracy', 'Precision', 'Recall', 
                      'F1-Score', 'ROC AUC', 'PR AUC', 'Inference Time (ms/sample)']
        simple_df = metrics_df[simple_cols].copy()
        simple_df.to_csv(os.path.join(RESULT_DIR, "dashboard_metrics.csv"), index=False)
        
        # Create combined visualizations
        plot_combined_metrics(simple_df)
        
        print("\n✅ All visualizations and metrics saved to:")
        print(f"- {RESULT_DIR}/full_metrics.csv")
        print(f"- {RESULT_DIR}/dashboard_metrics.csv")
        print(f"- {VISUAL_DIR}/ directory")
    else:
        print("⚠️ No metrics were generated due to errors")

if __name__ == "__main__":
    main()
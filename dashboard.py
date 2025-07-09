import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from PIL import Image
import glob
from sklearn.metrics import roc_curve, auc

# Configure the page
st.set_page_config(
    page_title="AI-Powered NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .comparison-table {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: #F0F2F6;
        border-radius: 10px 10px 0px 0px;
        gap: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    metrics = pd.read_csv("results/dashboard_metrics.csv")
    return metrics

def plot_dataset_comparison(metrics_df):
    """Create comparison plot for all datasets"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_metrics = metrics_df.groupby('Dataset').mean(numeric_only=True).reset_index()
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    
    for i, dataset in enumerate(avg_metrics['Dataset'].unique()):
        dataset_data = avg_metrics[avg_metrics['Dataset'] == dataset]
        values = [dataset_data[metric].values[0] for metric in metrics_to_plot]
        ax.bar(x + i*width, values, width, label=dataset)
    
    ax.set_ylabel('Score')
    ax.set_title('Average Performance by Dataset')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    for i, dataset in enumerate(avg_metrics['Dataset'].unique()):
        dataset_data = avg_metrics[avg_metrics['Dataset'] == dataset]
        for j, metric in enumerate(metrics_to_plot):
            value = dataset_data[metric].values[0]
            ax.text(j + i*width, value + 0.02, f"{value:.2f}", ha='center')
    
    plt.tight_layout()
    return fig

def plot_dataset_roc_comparison():
    """Create combined ROC curve plot for all datasets"""
    plt.figure(figsize=(10, 6))
    
    dataset_styles = {
        "CICIDS2017": {'color': '#3498db', 'linestyle': '-', 'label': 'CICIDS2017'},
        "UNSW-NB15": {'color': '#e74c3c', 'linestyle': '--', 'label': 'UNSW-NB15'},
        "DNS-Reflection-DDoS": {'color': '#2ecc71', 'linestyle': ':', 'label': 'DNS-Reflection'}
    }
    
    auc_stats = []
    
    for dataset, style in dataset_styles.items():
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        model_files = glob.glob(f"models/{dataset}_*.pkl")
        model_files = [f for f in model_files if "_meta" not in f]
        
        for model_file in model_files:
            try:
                model_name = os.path.basename(model_file).replace(f"{dataset}_", "").replace(".pkl", "")
                model = joblib.load(model_file)
                meta = joblib.load(f"models/{dataset}_{model_name}_meta.pkl")
                
                test_data = pd.read_csv(f"results/{dataset}_test.csv")
                X_test = test_data[meta['features']]
                y_test = test_data['Label']
                X_test_scaled = meta['scaler'].transform(X_test)
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test_scaled)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(auc(fpr, tpr))
            except Exception as e:
                print(f"Error processing {model_file}: {str(e)}")
                continue
        
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, 
                     color=style['color'],
                     linestyle=style['linestyle'],
                     linewidth=2,
                     label=f"{style['label']} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                            color=style['color'], alpha=0.2)
            auc_stats.append({
                'Dataset': style['label'],
                'Mean AUC': mean_auc,
                'Std Dev': std_auc
            })
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Dataset ROC Curve Comparison (Mean ± 1 SD)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    auc_df = pd.DataFrame(auc_stats) if auc_stats else pd.DataFrame()
    return plt.gcf(), auc_df

# ✅ New Function Added Below
def plot_model_roc_comparison(dataset):
    """Plot ROC curves for all models within a specific dataset"""
    plt.figure(figsize=(10, 6))
    model_files = glob.glob(f"models/{dataset}_*.pkl")
    model_files = [f for f in model_files if "_meta" not in f]
    
    for model_file in model_files:
        try:
            model_name = os.path.basename(model_file).replace(f"{dataset}_", "").replace(".pkl", "")
            model = joblib.load(model_file)
            meta = joblib.load(f"models/{dataset}_{model_name}_meta.pkl")
            test_data = pd.read_csv(f"results/{dataset}_test.csv")
            X_test = test_data[meta['features']]
            y_test = test_data['Label']
            X_test_scaled = meta['scaler'].transform(X_test)
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")
        except Exception as e:
            print(f"Error in {model_file}: {str(e)}")
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f'ROC Curve Comparison - {dataset}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def main():
    st.title("🛡️ AI-Powered Network Intrusion Detection System")
    st.markdown("---")
    
    metrics_df = load_data()
    st.sidebar.title("Navigation & Filters")
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=metrics_df['Dataset'].unique(),
        index=0
    )
    dataset_models = metrics_df[metrics_df['Dataset'] == selected_dataset]['Model'].unique()
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=dataset_models,
        index=0
    )
    model_data = metrics_df[(metrics_df['Dataset'] == selected_dataset) & 
                          (metrics_df['Model'] == selected_model)].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### 📊 {selected_model} Performance")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Accuracy", f"{model_data['Accuracy']:.2%}")
        st.metric("Precision", f"{model_data['Precision']:.2%}")
        st.metric("Recall", f"{model_data['Recall']:.2%}")
        st.metric("F1-Score", f"{model_data['F1-Score']:.2%}")
        st.metric("ROC AUC", f"{model_data['ROC AUC']:.2%}")
        st.metric("Inference Time", f"{model_data['Inference Time (ms/sample)']:.2f} ms")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"### 📂 Dataset: {selected_dataset}")
        st.write(f"Showing results for {selected_model} model")
        
    with col2:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Model ROC", 
            "🤖 Confusion Matrix", 
            "📋 Model Metrics",
            "📊 Dataset Comparison",
            "🌐 Dataset ROC Comparison"
        ])
        
        with tab1:
            try:
                roc_img = Image.open(f"results/visualizations/{selected_dataset}_{selected_model}_roc.png")
                st.image(roc_img, caption=f"ROC Curve - {selected_model}")
            except FileNotFoundError:
                st.warning("ROC curve image not found")
        
        with tab2:
            try:
                cm_img = Image.open(f"results/visualizations/{selected_dataset}_{selected_model}_confusion.png")
                st.image(cm_img, caption=f"Confusion Matrix - {selected_model}")
            except FileNotFoundError:
                st.warning("Confusion matrix image not found")
        
        with tab3:
            st.markdown("### All Model Metrics Comparison")
            dataset_metrics = metrics_df[metrics_df['Dataset'] == selected_dataset]
            st.dataframe(
                dataset_metrics.drop(columns=['Dataset']).set_index('Model'),
                use_container_width=True,
                height=300
            )
            try:
                combined_img = Image.open("results/visualizations/combined_metrics.png")
                st.image(combined_img, caption="All Models Performance Comparison")
            except FileNotFoundError:
                st.warning("Combined metrics image not found")
        
        with tab4:
            st.markdown("### Dataset Comparison")
            st.markdown("#### Average Performance Across Datasets")
            comparison_fig = plot_dataset_comparison(metrics_df)
            st.pyplot(comparison_fig)
            st.markdown("#### Detailed Metrics Comparison")
            st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
            pivot_metrics = metrics_df.pivot_table(
                index='Model',
                columns='Dataset',
                values=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                aggfunc='mean'
            )
            pivot_metrics.columns = [f"{col[1]} {col[0]}" for col in pivot_metrics.columns]
            st.dataframe(
                pivot_metrics.style.format("{:.2%}"),
                use_container_width=True,
                height=400
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown("### Combined ROC Curves by Dataset")
            roc_fig, auc_table = plot_dataset_roc_comparison()
            st.pyplot(roc_fig)
            if not auc_table.empty:
                st.markdown("#### AUC Statistics")
                st.dataframe(
                    auc_table.style.format({
                        'Mean AUC': '{:.3f}',
                        'Std Dev': '{:.3f}'
                    }).highlight_max(subset=['Mean AUC'], color='#2ecc71')
                    .highlight_min(subset=['Mean AUC'], color='#e74c3c'),
                    use_container_width=True
                )
            else:
                st.warning("Could not generate ROC comparison. Check if all model files exist.")
            
            # ✅ New section for individual model ROC curves
            st.markdown("### Model ROC Comparison Within Dataset")
            roc_model_fig = plot_model_roc_comparison(selected_dataset)
            st.pyplot(roc_model_fig)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d;">
        <p>AI-Powered NIDS Dashboard • Final Year Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

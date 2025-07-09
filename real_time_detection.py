import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from scapy.all import sniff, IP, TCP, UDP
import socket
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Configuration
MODEL_DIR = "models"
UPDATE_INTERVAL = 1  # seconds
PACKET_LIMIT = 500  # packets to capture per analysis window

# Available models and datasets
DATASETS = {
    "CICIDS2017": {
        "models": ["Random_Forest", "XGBoost", "Decision_Tree", "Logistic_Regression", "Neural_Network"],
        "test_file": "results/CICIDS2017_test.csv"
    }
}

# UI Configuration
st.set_page_config(
    page_title="AI-Powered NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .attack-detected {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .normal-traffic {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(dataset_name, model_name):
    model_path = f"{MODEL_DIR}/{dataset_name}_{model_name}.pkl"
    meta_path = f"{MODEL_DIR}/{dataset_name}_{model_name}_meta.pkl"
    
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    
    return model, meta['scaler'], meta['features']

def get_network_interfaces():
    """Get available network interfaces"""
    try:
        import netifaces
        return netifaces.interfaces()
    except:
        return ["eth0", "wlan0", "lo"]  # Default fallback

def get_local_ip():
    """Get actual local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except:
        return socket.gethostbyname(socket.gethostname())

def real_time_detection():
    st.title("🛡️ AI-Powered Network Intrusion Detection System")
    
    # Initialize session state
    if 'detection_stats' not in st.session_state:
        st.session_state.detection_stats = {
            'normal': 0,
            'attack': 0,
            'packets_processed': 0,
            'start_time': time.time()
        }
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if 'active_model' not in st.session_state:
        st.session_state.active_model = None

    # Model selection sidebar
    with st.sidebar:
        st.header("Configuration")
        dataset_name = st.selectbox("Select Dataset", list(DATASETS.keys()))
        model_name = st.selectbox("Select Model", DATASETS[dataset_name]["models"])
        
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                st.session_state.active_model = load_model(dataset_name, model_name)
                st.success(f"{model_name} model loaded!")
        
        st.markdown("---")
        st.header("Network Settings")
        interface = st.selectbox("Network Interface", get_network_interfaces(), index=0)
        st.info(f"Actual IP: {get_local_ip()}")

    # Main content
    tab1, tab2 = st.tabs(["📊 Simulation Mode", "🌐 Live Detection"])

    with tab1:
        st.header("Dataset Simulation Mode")
        
        if st.session_state.active_model is None:
            st.warning("Please load a model first from the sidebar")
        else:
            model, scaler, feature_columns = st.session_state.active_model
            test_data = pd.read_csv(DATASETS[dataset_name]["test_file"])
            test_features = test_data[feature_columns]
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Simulation", key="start_sim"):
                    st.session_state.sim_running = True
                    st.session_state.detection_stats = {
                        'normal': 0,
                        'attack': 0,
                        'packets_processed': 0,
                        'start_time': time.time()
                    }
                    st.session_state.results_df = pd.DataFrame()
            
            with col2:
                if st.button("Stop Simulation", key="stop_sim"):
                    st.session_state.sim_running = False
            
            if st.session_state.get('sim_running', False):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                for i in range(min(100, len(test_features))):  # Process first 100 samples
                    if not st.session_state.get('sim_running', False):
                        break
                    
                    # Process sample
                    sample = test_features.iloc[i].values.reshape(1, -1)
                    sample_scaled = scaler.transform(sample)
                    
                    # Predict
                    proba = model.predict_proba(sample_scaled)[0]
                    prediction = model.predict(sample_scaled)[0]
                    
                    # Update stats
                    result_type = 'attack' if prediction == 1 else 'normal'
                    st.session_state.detection_stats[result_type] += 1
                    st.session_state.detection_stats['packets_processed'] += 1
                    
                    # Store results
                    new_row = {
                        'Timestamp': pd.Timestamp.now(),
                        'Packet ID': i+1,
                        'Type': 'Attack' if prediction == 1 else 'Normal',
                        'Confidence': f"{proba[1 if prediction == 1 else 0]:.1%}",
                        'Risk Score': f"{proba[1]:.1%}",
                        'Model': model_name
                    }
                    st.session_state.results_df = pd.concat([
                        st.session_state.results_df,
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
                    
                    # Update UI
                    progress = (i+1) / min(100, len(test_features))
                    progress_bar.progress(progress)
                    status_text.markdown(f"""
                    <div class="metric-box {'attack-detected' if prediction == 1 else 'normal-traffic'}">
                        <h4>Packet {i+1}: {'🚨 ATTACK' if prediction == 1 else '✅ NORMAL'}</h4>
                        <p>Confidence: {proba[1 if prediction == 1 else 0]:.1%}</p>
                        <p>Risk Score: {proba[1]:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(0.3)  # Simulate processing delay
                
                st.session_state.sim_running = False
                st.rerun()
            
            # Show final results if available
            if not st.session_state.results_df.empty:
                st.subheader("Simulation Results Summary")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Packets", st.session_state.detection_stats['packets_processed'])
                col2.metric("Normal Packets", st.session_state.detection_stats['normal'])
                col3.metric("Attack Packets", st.session_state.detection_stats['attack'])
                
                st.dataframe(
                    st.session_state.results_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Plot results
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                
                # Pie chart
                ax[0].pie(
                    [st.session_state.detection_stats['normal'], st.session_state.detection_stats['attack']],
                    labels=['Normal', 'Attack'],
                    colors=['#4caf50', '#f44336'],
                    autopct='%1.1f%%'
                )
                ax[0].set_title('Traffic Distribution')
                
                # Risk score over time
                if len(st.session_state.results_df) > 1:
                    risk_scores = st.session_state.results_df['Risk Score'].str.rstrip('%').astype('float') / 100
                    ax[1].plot(
                        st.session_state.results_df['Packet ID'],
                        risk_scores,
                        'r-',
                        alpha=0.7
                    )
                    ax[1].set_xlabel('Packet Sequence')
                    ax[1].set_ylabel('Risk Score')
                    ax[1].set_ylim(0, 1)
                    ax[1].grid(True, alpha=0.3)
                    ax[1].set_title('Risk Score Over Time')
                
                st.pyplot(fig)

    with tab2:
        st.header("Live Network Detection")
        
        if st.session_state.active_model is None:
            st.warning("Please load a model first from the sidebar")
        else:
            model, scaler, feature_columns = st.session_state.active_model
            
            if st.button("Start Live Capture", key="start_live"):
                st.session_state.live_running = True
                st.session_state.detection_stats = {
                    'normal': 0,
                    'attack': 0,
                    'packets_processed': 0,
                    'start_time': time.time()
                }
                st.session_state.results_df = pd.DataFrame()
                
                # Packet processing function
                def process_packet(packet):
                    if not st.session_state.get('live_running', False):
                        return
                    
                    try:
                        features = extract_packet_features(packet)
                        df = pd.DataFrame([features])[feature_columns]
                        
                        if not df.empty:
                            df_scaled = scaler.transform(df)
                            proba = model.predict_proba(df_scaled)[0]
                            prediction = model.predict(df_scaled)[0]
                            
                            # Update stats
                            result_type = 'attack' if prediction == 1 else 'normal'
                            st.session_state.detection_stats[result_type] += 1
                            st.session_state.detection_stats['packets_processed'] += 1
                            
                            # Store results
                            new_row = {
                                'Timestamp': pd.Timestamp.now(),
                                'Source IP': packet[IP].src if IP in packet else 'N/A',
                                'Destination IP': packet[IP].dst if IP in packet else 'N/A',
                                'Protocol': packet[IP].proto if IP in packet else 'N/A',
                                'Type': 'Attack' if prediction == 1 else 'Normal',
                                'Confidence': f"{proba[1 if prediction == 1 else 0]:.1%}",
                                'Risk Score': f"{proba[1]:.1%}",
                                'Model': model_name
                            }
                            st.session_state.results_df = pd.concat([
                                st.session_state.results_df,
                                pd.DataFrame([new_row])
                            ], ignore_index=True)
                    except Exception as e:
                        print(f"Error processing packet: {e}")
                
                # Start capture in background
                from threading import Thread
                capture_thread = Thread(
                    target=lambda: sniff(prn=process_packet, store=0),
                    daemon=True
                )
                capture_thread.start()
            
            if st.button("Stop Capture", key="stop_live"):
                st.session_state.live_running = False
            
            if st.session_state.get('live_running', False):
                status_box = st.empty()
                stats_box = st.empty()
                results_box = st.empty()
                
                while st.session_state.get('live_running', False):
                    with status_box.container():
                        st.success("🟢 Live capture running...")
                        
                        # Show latest result if available
                        if not st.session_state.results_df.empty:
                            latest = st.session_state.results_df.iloc[-1]
                            alert_type = "attack-detected" if latest['Type'] == 'Attack' else "normal-traffic"
                            st.markdown(f"""
                            <div class="metric-box {alert_type}">
                                <h4>Latest Packet: {'🚨 ATTACK' if latest['Type'] == 'Attack' else '✅ NORMAL'}</h4>
                                <p>Source: {latest['Source IP']} → Dest: {latest['Destination IP']}</p>
                                <p>Protocol: {latest['Protocol']} | Confidence: {latest['Confidence']}</p>
                                <p>Risk Score: {latest['Risk Score']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with stats_box.container():
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Packets", st.session_state.detection_stats['packets_processed'])
                        col2.metric("Normal Packets", st.session_state.detection_stats['normal'])
                        col3.metric("Attack Packets", st.session_state.detection_stats['attack'])
                        
                        # Plot live data
                        if len(st.session_state.results_df) > 1:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            risk_scores = st.session_state.results_df['Risk Score'].str.rstrip('%').astype('float') / 100
                            ax.plot(
                                st.session_state.results_df['Timestamp'],
                                risk_scores,
                                'r-',
                                alpha=0.7
                            )
                            ax.set_xlabel('Time')
                            ax.set_ylabel('Risk Score')
                            ax.set_ylim(0, 1)
                            ax.grid(True, alpha=0.3)
                            ax.set_title('Live Risk Score Monitoring')
                            st.pyplot(fig)
                    
                    time.sleep(UPDATE_INTERVAL)
                
                st.rerun()
            
            # Show final results if available
            if not st.session_state.results_df.empty and not st.session_state.get('live_running', False):
                st.subheader("Capture Results Summary")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Packets", st.session_state.detection_stats['packets_processed'])
                col2.metric("Normal Packets", st.session_state.detection_stats['normal'])
                col3.metric("Attack Packets", st.session_state.detection_stats['attack'])
                
                st.dataframe(
                    st.session_state.results_df,
                    use_container_width=True,
                    hide_index=True
                )

def main():
    real_time_detection()

if __name__ == "__main__":
    main()
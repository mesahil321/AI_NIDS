import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_dataset(df):
    """Handle infinite/NaN/extreme values"""
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e10, upper=1e10)
    return df

# ===== UNSW-NB15 (UNCHANGED) =====
def preprocess_unsw(df_unsw):
    print("\nProcessing UNSW-NB15 dataset...")
    df_unsw.columns = df_unsw.columns.str.strip()
    df_unsw = clean_dataset(df_unsw)
    df_unsw['Label'] = df_unsw['label'].apply(lambda x: 1 if x != 0 else 0)
    
    categorical_cols = df_unsw.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df_unsw[col] = le.fit_transform(df_unsw[col].astype(str))
    
    output_path = r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\UNSW-NB15\cleaned_unsw_nb15.csv"
    df_unsw.to_csv(output_path, index=False)
    print(f"UNSW-NB15 preprocessing complete! Saved to {output_path}")
    return df_unsw

# ===== CICIDS2017 (FIXED) =====
def preprocess_cicids(df_cicids):
    print("\nProcessing CICIDS2017 dataset...")
    df_cicids.columns = df_cicids.columns.str.strip()
    df_cicids = clean_dataset(df_cicids)
    df_cicids['Label'] = df_cicids['Label'].apply(
        lambda x: 1 if str(x).strip().upper() != 'BENIGN' else 0)
    
    # Remove leaky features
    leaky_features = ['Flow Bytes/s', 'Flow Packets/s', 'Packet Length Std']
    df_cicids = df_cicids.drop(columns=leaky_features, errors='ignore')
    
    # Safe downsampling
    benign = df_cicids[df_cicids['Label'] == 0]
    attack = df_cicids[df_cicids['Label'] == 1]
    
    if len(benign) > len(attack):
        benign = benign.sample(n=len(attack), random_state=42)
    elif len(attack) > len(benign):
        attack = attack.sample(n=len(benign), random_state=42)
    
    df_cicids = pd.concat([benign, attack])
    
    output_path = r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\CICISD-2017\cleaned_cicids2017.csv"
    df_cicids.to_csv(output_path, index=False)
    print(f"CICIDS2017 preprocessing complete! Saved to {output_path}")
    return df_cicids

# ===== DNS-Reflection-DDoS (FIXED) =====
def preprocess_ddos(df_ddos):
    print("\nProcessing DNS-Reflection-DDoS dataset...")
    # Handle mixed dtype warning
    df_ddos = pd.read_csv(r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\DrDoS_DNS\DrDoS_DNS.csv", low_memory=False)
    df_ddos.columns = df_ddos.columns.str.strip()
    
    label_col = 'Label' if 'Label' in df_ddos.columns else 'label'
    df_ddos = clean_dataset(df_ddos)
    df_ddos['Label'] = df_ddos[label_col].apply(
        lambda x: 0 if str(x).strip().upper() in ['BENIGN', 'NORMAL'] else 1)
    
    # Remove high-correlation features
    df_ddos = df_ddos.drop(columns=['Flow Bytes/s', 'Flow Packets/s'], errors='ignore')
    
    # Safe balancing
    normal = df_ddos[df_ddos['Label'] == 0]
    attack = df_ddos[df_ddos['Label'] == 1]
    
    if len(normal) > len(attack):
        normal = normal.sample(n=len(attack), random_state=42)
    elif len(attack) > len(normal):
        attack = attack.sample(n=len(normal), random_state=42)
    
    df_ddos = pd.concat([normal, attack])
    
    # Encode categoricals
    categorical_cols = df_ddos.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df_ddos[col] = le.fit_transform(df_ddos[col].astype(str))
    
    output_path = r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\DrDoS_DNS\cleaned_drdos_dns.csv"
    df_ddos.to_csv(output_path, index=False)
    print(f"DNS-Reflection-DDoS preprocessing complete! Saved to {output_path}")
    return df_ddos

if __name__ == "__main__":
    try:
        # Load data with error handling
        print("Loading datasets...")
        df_unsw = pd.read_csv(r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\UNSW-NB15\UNSW_NB15_training-set.csv")
        df_cicids = pd.read_csv(r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\CICISD-2017\Friday-WorkingHours-Afternoon-DDos.csv")
        df_ddos = pd.read_csv(r"C:\Users\parve\OneDrive\Desktop\IDS_Dataset\DrDoS_DNS\DrDoS_DNS.csv", low_memory=False)
        
        # Process datasets
        preprocess_unsw(df_unsw)
        preprocess_cicids(df_cicids)
        preprocess_ddos(df_ddos)
        
        print("\nAll datasets processed successfully!")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("Debugging Info:")
        print("1. Verify these files exist:")
        print(r" - C:\Users\parve\OneDrive\Desktop\IDS_Dataset\UNSW-NB15\UNSW_NB15_training-set.csv")
        print(r" - C:\Users\parve\OneDrive\Desktop\IDS_Dataset\CICISD-2017\Friday-WorkingHours-Afternoon-DDos.csv")
        print(r" - C:\Users\parve\OneDrive\Desktop\IDS_Dataset\DrDoS_DNS\DrDoS_DNS.csv")
        print("\n2. Check file permissions")
        print("3. Ensure sufficient disk space")
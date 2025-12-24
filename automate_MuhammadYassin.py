import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Menghapus data duplikat
    df = df.drop_duplicates()

    # Menangani missing value pada kolom TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Encoding data kategorikal
    categorical_cols = df.select_dtypes(include=["object"]).columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # Standarisasi fitur numerik
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Memisahkan fitur dan target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y

if __name__ == "__main__":
    df = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X, y = preprocess_data(df)

    print("Preprocessing selesai")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

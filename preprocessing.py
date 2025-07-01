from sklearn.preprocessing import LabelEncoder, StandardScaler
def preprocess(df):
    # Assume you already have a list of categorical columns
    categorical_cols = ['Education', 'Securities Account', 'CD Account', 'Online', 'CreditCard', 'ZIP Code']
    
    
    # Get numeric (non-categorical) columns
    numeric_cols = [col for col in df.columns if col not in categorical_cols + ['ID', 'Personal Loan']]
    
    # Label Encode Categorical Columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoder if needed later (e.g., inverse_transform)
    
    # Standard Scale Numeric Columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Now `df` has all categorical columns label encoded and numeric columns scaled
    print("Transformed DataFrame:")
    print(df.head())
    return df
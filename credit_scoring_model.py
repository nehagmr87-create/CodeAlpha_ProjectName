import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic dataset for demonstration
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'income': np.random.normal(60000, 20000, n_samples),
        'debt': np.random.normal(20000, 10000, n_samples),
        'credit_history': np.random.randint(1, 10, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'creditworthy': np.zeros(n_samples)
    }
    
    # Simple rule for creditworthiness (can be modified)
    for i in range(n_samples):
        score = (0.4 * (data['income'][i]/100000) - 
                0.3 * (data['debt'][i]/50000) + 
                0.3 * (data['credit_history'][i]/10))
        data['creditworthy'][i] = 1 if score > 0.5 else 0
    
    return pd.DataFrame(data)

# Train credit scoring model
def train_model(df):
    X = df[['income', 'debt', 'credit_history', 'age']]
    y = df['creditworthy']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

# Predict creditworthiness for new applicant
def predict_creditworthiness(model, scaler):
    print("\nEnter Applicant Details:")
    try:
        income = float(input("Annual Income ($): "))
        debt = float(input("Total Debt ($): "))
        credit_history = float(input("Credit History Score (1-10): "))
        age = float(input("Age: "))
        
        # Prepare input data
        input_data = np.array([[income, debt, credit_history, age]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Display result
        result = "Creditworthy" if prediction[0] == 1 else "Not Creditworthy"
        print(f"\nPrediction: {result}")
        print(f"Creditworthy Probability: {probability:.2%}")
        
    except ValueError:
        print("Please enter valid numerical values.")

# Main execution
if __name__ == "__main__": # Changed _name_ to __name__
    # Generate and prepare data
    df = generate_synthetic_data()
    print("Dataset generated successfully.")
    
    # Train model
    model, scaler = train_model(df)
    
    # Get user input and predict
    while True:
        predict_creditworthiness(model, scaler)
        again = input("\nWould you like to evaluate another applicant? (y/n): ").lower()
        if again != 'y':
            break
    
    print("Thank you for using the Credit Scoring Model!")
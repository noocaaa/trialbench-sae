import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(phase):
    # Load data for the specified phase
    train_df = pd.read_csv(f'data/serious-adverse-event-forecasting/{phase}/train_x.csv')
    test_df = pd.read_csv(f'data/serious-adverse-event-forecasting/{phase}/test_x.csv')
    
    # Split train into train/validation
    train, valid = train_test_split(train_df, test_size=0.2, random_state=42)
    
    return train, valid, test_df

# Example usage with Phase1
train, valid, test_df = load_data('Phase1')

print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test_df)}")
print(f"Features: {train.shape[1]}")

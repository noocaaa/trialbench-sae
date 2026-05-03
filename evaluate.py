import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data(phase, data_root=None):
    """Load Phase data from the absolute data directory path."""
    if data_root is None:
        data_root = Path(__file__).resolve().parent / 'data'
    else:
        data_root = Path(data_root).expanduser().resolve()

    phase_dir = data_root / 'serious-adverse-event-forecasting' / phase
    train_path = phase_dir / 'train_x.csv'
    test_path = phase_dir / 'test_x.csv'

    print(f'Loading data from: {train_path}')
    print(f'Loading data from: {test_path}')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train, valid = train_test_split(train_df, test_size=0.2, random_state=42)
    return train, valid, test_df


if __name__ == '__main__':
    train, valid, test_df = load_data('Phase1')
    print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test_df)}")
    print(f"Features: {train.shape[1]}")

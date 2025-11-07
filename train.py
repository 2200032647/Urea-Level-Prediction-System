import argparse
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['urea_level'])
    # one-hot encode crop_type
    X = pd.get_dummies(X, columns=['crop_type'], prefix='crop')
    y = df['urea_level']
    return X, y

def train(data_path, out_model_path):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Test RMSE: {rmse:.3f} (kg/ha)")
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    with open(out_model_path, 'wb') as f:
        pickle.dump({'model': model, 'columns': X.columns.tolist()}, f)
    print(f"Model saved to {out_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    train(args.data, args.out)

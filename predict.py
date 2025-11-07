import argparse
import pandas as pd
import pickle
import numpy as np

def load_model(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data['model'], data['columns']

def prepare_input(df, columns):
    df2 = pd.get_dummies(df, columns=['crop_type'], prefix='crop')
    # ensure columns
    for c in columns:
        if c not in df2.columns:
            df2[c] = 0
    df2 = df2[columns]
    return df2

def predict(model_path, input_csv, out_csv):
    model, columns = load_model(model_path)
    df = pd.read_csv(input_csv)
    X = prepare_input(df, columns)
    preds = model.predict(X)
    df_out = df.copy()
    df_out['predicted_urea_level_kg_per_ha'] = preds
    df_out.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    predict(args.model, args.input, args.output)

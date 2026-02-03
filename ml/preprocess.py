import pandas as pd
import numpy as np
import os

# Define column names based on CMAPSS documentation
index_names = ['unit', 'cycles']
setting_names = ['setting1', 'setting2', 'setting3']
sensor_names = ['s' + str(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    # Load space-separated file
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    return df

def add_rul(df):
    # Calculate Remaining Useful Life (RUL)
    # 1. Group by unit and find the maximum cycle for each unit
    max_cycles = df.groupby('unit')['cycles'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    
    # 2. Merge back to original dataframe
    df = df.merge(max_cycles, on='unit', how='left')
    
    # 3. RUL = max_cycle - current_cycle
    df['RUL'] = df['max_cycle'] - df['cycles']
    
    # Drop max_cycle as it was just a helper
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def preprocess_and_save():
    input_file = 'data/train_FD001.txt'
    output_file = 'data/processed_train_FD001.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run download script first.")
        return

    # Load
    df = load_data(input_file)
    
    # Add target (RUL)
    df = add_rul(df)
    
    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    preprocess_and_save()

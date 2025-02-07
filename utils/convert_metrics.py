import sys
import pandas as pd
import os

def convert_metrics(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Select only rows from P_2 to map_cut_10
    metrics_to_keep = [
        'P_2', 'P_5', 'P_10',
        'recall_2', 'recall_5', 'recall_10',
        'ndcg_cut_2', 'ndcg_cut_5', 'ndcg_cut_10',
        'map_cut_2', 'map_cut_5', 'map_cut_10'
    ]
    
    # Assuming first column contains metric names
    # Set it as index before filtering
    df = df.set_index(df.columns[0])
    
    # Filter rows and get only the 'mean' column
    filtered_df = df.loc[df.index.isin(metrics_to_keep), ['mean']]
    
    # Multiply values by 100 to move decimal point right 2 places
    filtered_df['mean'] = filtered_df['mean'] * 100
    
    # Generate output filename
    file_path, file_extension = os.path.splitext(input_file)
    output_file = f"{file_path}_converted{file_extension}"
    
    # Convert to single row string and write without newline
    values = filtered_df['mean'].values
    row = ','.join(f"{x}" for x in values)
    with open(output_file, 'w') as f:
        f.write(row)
    
    print(f"Converted metrics saved to: {output_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 convert_metrics.py path_to_csv_file")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
        
    convert_metrics(input_file)

if __name__ == "__main__":
    main()

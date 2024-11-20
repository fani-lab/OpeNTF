import os
import pickle
import logging
import sys
import argparse
from scipy.sparse import isspmatrix

def setup_logger(log_path):
    """Setup logger to log to a file."""
    logger = logging.getLogger("PklReaderLogger")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # Create formatter and add to handler
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(fh)
    return logger

def read_pkl_file(pkl_path):
    """Read a .pkl file and return its contents."""
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    return data

def print_rows(data, logger, num_rows):
    """Print the specified number of rows or key-value pairs of the data and log them."""
    if isinstance(data, (list, tuple)):
        rows = data[:num_rows]
    elif isinstance(data, dict):  # Check if data is a dictionary
        rows = list(data.items())[:num_rows]
    elif hasattr(data, 'head'):  # Check if data is a DataFrame
        rows = data.head(num_rows)
    elif isspmatrix(data):  # Check if data is a sparse matrix
        rows = data[:num_rows].toarray()  # Convert the specified number of rows to a dense format for printing
    else:
        logger.info("Unsupported data format.")
        print("Unsupported data format:", type(data))
        sys.exit("Unsupported data format.")
    
    for i, row in enumerate(rows, 1):
        if isinstance(row, tuple) and isspmatrix(row[1]):
            # For sparse matrix values in a dictionary, show the specified number of rows in dense form
            logger.info(f"Row {i} Key: {row[0]}")
            print(f"Row {i} Key: {row[0]}")
            logger.info(f"First {num_rows} rows of sparse matrix:\n{row[1][:num_rows].toarray()}")
            print(f"First {num_rows} rows of sparse matrix:\n{row[1][:num_rows].toarray()}")
        else:
            logger.info(f"Row {i}: {row}")
            print(f"Row {i}: {row}")

if __name__ == "__main__":
    # Parse the number of rows to read from command-line arguments
    parser = argparse.ArgumentParser(description="Read and print specified number of rows from teamsvecs.pkl")
    parser.add_argument("num_rows", type=int, help="Number of rows to read and print")
    args = parser.parse_args()

    # Hardcoded .pkl file path
    pkl_path = "teamsvecs.pkl"
    
    # Define the .log file path with the .pkl filename + "_output.log"
    base_name = os.path.splitext(os.path.basename(pkl_path))[0]
    log_path = os.path.join(os.path.dirname(pkl_path), f"{base_name}_output.log")
    
    # Setup logger
    logger = setup_logger(log_path)
    
    # Read and print specified number of rows from the .pkl file
    data = read_pkl_file(pkl_path)
    print_rows(data, logger, args.num_rows)

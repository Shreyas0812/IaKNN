import pandas as pd
import numpy as np

def load_data(file_path, chunk_size=1000, selected_columns=None):
    """
    Load data from a CSV file in chunks.

    Parameters:
    - file_path: str, path to the CSV file
    - chunk_size: int, number of rows per chunk
    - selected_columns: list, columns to select from the CSV

    Returns:
    - generator of DataFrames
    """
    # Read the CSV file in chunks
    chunk_iter = pd.read_csv(file_path, usecols=selected_columns, chunksize=chunk_size)

    # Process each chunk
    for chunk in chunk_iter:
        yield chunk


if __name__ == "__main__":

    file_path = 'products-10000.csv'
    chunk_size = 1000

    selected_columns = ['Index', 'Name', 'Color', 'Size', 'Price']

    # Load data in chunks
    for chunk in load_data(file_path, chunk_size, selected_columns):
        # Process each chunk
        print(chunk.head()) 

        state_matrix = chunk.to_numpy()

        print(state_matrix.shape)
        print()

 
        
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

    if selected_columns is None:
        chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
    else:
        chunk_iter = pd.read_csv(file_path, usecols=selected_columns, chunksize=chunk_size)

    # Process each chunk
    for chunk in chunk_iter:
        yield chunk

if __name__ == "__main__":

    chunk_data = True
    chunk_size = 10000
    num_chunks = 10
    selected_columns = None

    if chunk_data:
        trajectories_data = pd.DataFrame()
        cur_chunk = 0
        for chunk in load_data('../data/ngsim.csv', chunk_size=chunk_size, selected_columns=selected_columns):
            # Process the chunk (e.g., append to a list or DataFrame)
            trajectories_data = pd.concat([trajectories_data, chunk], ignore_index=True)

            cur_chunk += 1
            if cur_chunk >= num_chunks:
                break
    else:
        trajectories_data = pd.read_csv('../data/ngsim.csv')  



def find_vehicles_with_continuous_frames(df, min_frames=70):
    result = []
    for vehicle_id, group in df.groupby('Vehicle_ID'):
        frames = group['Frame_ID'].sort_values().values
        diffs = np.diff(frames)
        # Find runs of consecutive frames
        run_starts = np.where(diffs != 1)[0] + 1
        runs = np.split(frames, run_starts)
        for run in runs:
            if len(run) >= min_frames:
                result.append((vehicle_id, run))
    return result

    # # Initialize results list
# results = []

# # Iterate through the filtered frames
# for i in range(len(us_101_data_grouped_by_frame) - 69):
#     # Get a window of 70 continuous frames
#     window = us_101_data_grouped_by_frame.iloc[i:i + 70]
    
#     # Check if all 70 frames have the same Vehicle_ID values
#     if all(set(window['Vehicle_ID'].iloc[0]).issubset(vehicle_ids) for vehicle_ids in window['Vehicle_ID']):
#         results.append((window.index[0], window.index.values))

# # Display the results
# results


veh5frame70data = []

for start in range(len(us_101_data_grouped_by_frame_filtered) - num_frames + 1):
    window = us_101_data_grouped_by_frame_filtered.iloc[start:start + num_frames]
    sets = window['Vehicle_ID'].apply(set).tolist()
    if sets:
        common_ids = set.intersection(*sets)
        if len(common_ids) >= num_vehicles:
            # Union of all index lists
            index_union = set().union(*window['index'].tolist())
            veh5frame70data.append({
                'Frame_ID': window['Frame_ID'].tolist(),
                'intersections': common_ids,
                'index': list(index_union)
            })
    else:
        index_union = set().union(*window['index'].tolist())
        veh5frame70data.append({
            'Frame_ID': window['Frame_ID'].tolist(),
            'intersections': set(),
            'index': list(index_union)
        })

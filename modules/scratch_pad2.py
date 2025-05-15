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


# veh5frame70data = []

# for start in range(len(us_101_data_grouped_by_frame_filtered) - num_frames + 1):
#     window = us_101_data_grouped_by_frame_filtered.iloc[start:start + num_frames]
#     sets = window['Vehicle_ID'].apply(set).tolist()
#     if sets:
#         common_ids = set.intersection(*sets)
#         if len(common_ids) >= num_vehicles:
#             # Union of all index lists
#             index_union = set().union(*window['index'].tolist())
#             veh5frame70data.append({
#                 'Frame_ID': window['Frame_ID'].tolist(),
#                 'intersections': common_ids,
#                 'index': list(index_union)
#             })
#     else:
#         index_union = set().union(*window['index'].tolist())
#         veh5frame70data.append({
#             'Frame_ID': window['Frame_ID'].tolist(),
#             'intersections': set(),
#             'index': list(index_union)
#         })



us_101_data_lanes_exploded = us_101_data_lanes.explode('index')

potential_egos = []

for start in range(len(us_101_data_grouped_by_frame_filtered) - num_frames + 1):
    window = us_101_data_grouped_by_frame_filtered.iloc[start:start + num_frames]
    sets = window['Vehicle_ID'].apply(set).tolist()
    if sets:
        common_ids = set.intersection(*sets)
        index_union = [i for sublist in window['index'] for i in (sublist if isinstance(sublist, list) else [sublist])]
        potential_egos.append({
            'Frame_ID': window['Frame_ID'].tolist(),
            'intersections': common_ids,
            'index': list(index_union)
        })


for potential_ego in potential_egos[100:101]:
    filtered = us_101_data_lanes_exploded[
        us_101_data_lanes_exploded['index'].isin(potential_ego['index'])
        & us_101_data_lanes_exploded['Vehicle_ID'].isin(list(potential_ego['intersections']))
        & us_101_data_lanes_exploded['Frame_ID'].isin(potential_ego['Frame_ID'])
    ]
    print(filtered)


for potential_ego in potential_egos[100:101]:
    idx_set = set(potential_ego['index'])
    filtered = us_101_data_lanes[
        us_101_data_lanes['index'].apply(
            lambda idxs: any(i in idx_set for i in (idxs if isinstance(idxs, (list, np.ndarray)) else [idxs]))
        )
        & us_101_data_lanes['Vehicle_ID'].isin(list(potential_ego['intersections']))
        & us_101_data_lanes['Frame_ID'].isin(potential_ego['Frame_ID'])
    ]
    print(filtered)



import numpy as np

ego_vehicle_id = 8

# For each frame, find the ego vehicle's position
ego_df = df[df['Vehicle_ID'] == ego_vehicle_id]

if not ego_df.empty:
    ego_x = ego_df.iloc[0]['Local_X']
    ego_y = ego_df.iloc[0]['Local_Y']
    # Calculate Euclidean distance for all vehicles to ego
    df['d'] = np.sqrt((df['Local_X'] - ego_x)**2 + (df['Local_Y'] - ego_y)**2)
else:
    df['d'] = np.nan




import numpy as np

def calculate_distances(df):
    df = df.copy()
    df['d'] = np.nan
    for frame_id in df['Frame_ID'].unique():
        frame_data = df[df['Frame_ID'] == frame_id]
        ego_vehicle = frame_data[frame_data['Vehicle_ID'] == 8]
        if not ego_vehicle.empty:
            ego_x = ego_vehicle.iloc[0]['Local_X']
            ego_y = ego_vehicle.iloc[0]['Local_Y']
            mask = df['Frame_ID'] == frame_id
            df.loc[mask, 'd'] = np.sqrt((df.loc[mask, 'Local_X'] - ego_x)**2 +
                                        (df.loc[mask, 'Local_Y'] - ego_y)**2)
    return df

result_df = calculate_distances(df)


Potential ego vehicle IDs: [ 8 21 25 34 40]
     Vehicle_ID           d  Frame_ID  Lane_ID  Preceding  Following
0             8   72.093286       129        4          5         21
1            18   35.488004       129        5         14         31
2            20   86.363773       129        3          9         47
3            21    0.000000       129        4          8         25
4            25   99.738175       129        4         21         34
..          ...         ...       ...      ...        ...        ...
145           8   65.849486       158        4          5         21
146          20   74.340334       158        3          9         47
147          21    0.000000       158        4          8         25
148          25   93.565290       158        4         21         34
149          31  102.540056       158        5         18         35

[150 rows x 6 columns]
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from models_approach2 import IaKNN

class PickleDataset(Dataset):
    def __init__(self, pickle_dir):
        self.files = [os.path.join(pickle_dir, f) 
                      for f in os.listdir(pickle_dir) if f.endswith('.pk1')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_pickle(self.files[idx])
        # Convert the DataFrame to a torch tensor (float32)
        tensor = torch.tensor(df.values, dtype=torch.float32)
        return tensor
    

if __name__ == "__main__":
    
    pickle_dir = '../data/ego_df/us-101'
    dataset = PickleDataset(pickle_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)


    # Example: iterate through one batch
    for batch in dataloader:
        # batch is a tensor of shape (batch_size, rows, columns)
        print(batch.shape)
        # send batch to your model
        # output = model(batch)
        break

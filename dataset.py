import torch
import torch.utils.data
import numpy as np
import pickle

def get_dataset(pkl_path):
    with open(pkl_path, 'rb') as file:
        data_dict = pickle.load(file)
    X = torch.utils.data.dataloader.DataLoader(Dataset(data_dict['x']))
    T = torch.utils.data.dataloader.DataLoader(Dataset(data_dict['t']))
    return X, T

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, content):
        self.content = content

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.content[idx]


if __name__ == "__main__":
    X, _ = get_dataset('train.pkl')
    for idx, ctt in enumerate(X):
        if idx > 5:
            break
        print(ctt)
    
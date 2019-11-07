import torch
import torch.utils.data
import numpy as np
import pickle

def get_dataset(pkl_path, batch_size=1, data_size=-1, offset=0):
    with open(pkl_path, 'rb') as file:
        data_dict = pickle.load(file)
    X = torch.utils.data.dataloader.DataLoader(dataset=SingleDataset(data_dict['x'], data_size, offset), 
                                               batch_size=batch_size,
                                               shuffle=True)
    T = torch.utils.data.dataloader.DataLoader(dataset=SingleDataset(data_dict['t'], data_size, offset), 
                                               batch_size=batch_size,
                                               shuffle=True)
    return X, T


def get_tutor_dataset(pkl_path, batch_size=1, data_size=-1, offset=0):
    with open(pkl_path, 'rb') as file:
        data_dict = pickle.load(file)
    X = torch.utils.data.dataloader.DataLoader(dataset=TutorDataset(data_dict, data_size, offset), 
                                               batch_size=batch_size,
                                               shuffle=True)
    return X


class SingleDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, content, data_size=-1, offset=0):
        if data_size <= 0:
            self.content = content
        else:
            self.content = content[offset: offset+data_size]

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.content[idx]


class TutorDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, content, data_size=-1, offset=0):
        if data_size <= 0:
            self.content = content
        else:
            self.content = {'x': content['x'][offset: offset+data_size],
                            't': content['t'][offset: offset+data_size]}

    def __len__(self):
        return len(self.content['x'])

    def __getitem__(self, idx):
        return self.content['x'][idx], self.content['t'][idx]


if __name__ == "__main__":
    X = get_tutor_dataset('train.pkl',1)
    Y = get_tutor_dataset('eval.pkl',1)
    print(len(X))
    print(len(Y))
    
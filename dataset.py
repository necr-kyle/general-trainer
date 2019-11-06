import torch

def load_data(filename, size=None, skip=0):
    u = np.loadtxt(filename, dtype=int, delimiter=',',skiprows=skip, max_rows=size)
    v = torch.from_numpy(u).to(torch.int64)
    return v

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, size=None, skip=0):
        self.content = load_data(csv_file, size, skip)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.content[idx]


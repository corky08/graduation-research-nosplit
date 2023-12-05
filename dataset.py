import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, feature, target, transform=None):
        self.transform = transform
        self.feature= feature
        self.data_num = len(feature)
        self.target = target

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
          out_feature = self.transform(self.feature)[0][idx]
          out_target = self.target[idx]
        else:
          out_feature = self.feature[idx]
          out_target =  self.target[idx]

        return out_feature, out_target

def make_data(data, batch_size) :
    if type(data).__name__ == "NonRCT":
      prospenty_feature, prospenty_target, feature, target = data.gen_data()
      prospenty_feature = torch.tensor(prospenty_feature, dtype=torch.float32)
      prospenty_target = torch.tensor(prospenty_target, dtype=torch.float32)
      feature = torch.tensor(feature, dtype=torch.float32)
      target = torch.tensor(target, dtype=torch.float32)
      prospenty_data = MyDataset(prospenty_feature, prospenty_target)
      train_data = MyDataset(feature, target)
      prospenty_loader = DataLoader(prospenty_data, batch_size=batch_size, shuffle=True)
      train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
      return prospenty_loader, train_loader
    else :
      feature, target = data.gen_data()
      feature = torch.tensor(feature, dtype=torch.float32)
      target = torch.tensor(target, dtype=torch.float32)
      train_data = MyDataset(feature, target)
      train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
      return train_loader


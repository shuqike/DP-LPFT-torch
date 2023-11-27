'''Refer to
https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
https://www.kaggle.com/code/abhiswain/pytorch-mnist-using-pretrained-resnet50
'''
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import Dataset,ConcatDataset,DataLoader
from torchvision import transforms


class MNIST_data(Dataset):
    """MNIST data set, one/three channel version"""
    
    def __init__(self, file_path, 
                 transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
                     transforms.Normalize(mean=(0.5,), std=(0.5,))])
                ):
        
        df = pd.read_csv(file_path)
        
        if len(df.columns) == 784:
            # test data
            # self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None] # three channel version
            self.X = df.values.reshape((-1,28,28)).astype(np.uint8) # one channel version
            self.y = None
        else:
            # training data
            # self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None] # three channel version
            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)
            self.y = torch.from_numpy(df.iloc[:,0].values)
            
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])


def get_dataloader(batch_size):
    train_dataset = MNIST_data('data/mnist_train.csv', transform= transforms.Compose(
                                [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    test_dataset = MNIST_data('data/mnist_test.csv')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

'''Refer to
https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
https://www.kaggle.com/code/abhiswain/pytorch-mnist-using-pretrained-resnet50
'''
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import Dataset,ConcatDataset,DataLoader
from torchvision import transforms


class ResumableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1]

    def __len__(self):
        return self.num_samples
    
    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])


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


def get_sampler():
    train_dataset = MNIST_data('data/mnist_train.csv', transform= transforms.Compose(
                                [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    return ResumableRandomSampler(train_dataset)


def get_dataloader(batch_size, sampler):
    train_dataset = MNIST_data('data/mnist_train.csv', transform= transforms.Compose(
                                [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    test_dataset = MNIST_data('data/mnist_test.csv')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler,
                                               shuffle=False,
                                               pin_memory=True,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return sampler, train_loader, test_loader

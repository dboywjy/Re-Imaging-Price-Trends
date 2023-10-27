import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op

from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split
from torch.utils import data 
from torch.autograd import Variable 
import torch
import torch.nn as nn
from torch.nn import init

import torch
from torchvision import datasets,transforms
import torch.utils.data as Dataset

from sklearn.model_selection import train_test_split

import config

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)

class CNNDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    def __len__(self):
        return len(self.Data)
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        
        return data, label

class DataLoad():
    
    def __init__(self, IMAGE_WIDTH, IMAGE_HEIGHT, train_val_years, test_years, target, path, batch_size):
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.train_val_years = train_val_years
        self.test_years = test_years
        self.target = target
        self.path= path
        self.batch_size = batch_size
        
    def processing_data(self):
        images_train = []
        for year in self.train_val_years:
            images_temp = np.memmap(op.join(self.path, f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8, mode='r').reshape(
                                (-1, self.IMAGE_HEIGHT[20], self.IMAGE_WIDTH[20]))
            images_train.append(images_temp)
        images_train = np.concatenate(images_train)

        images_test = []
        for year in self.test_years:
            images_temp = np.memmap(op.join(self.path, f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8, mode='r').reshape(
                                (-1, self.IMAGE_HEIGHT[20], self.IMAGE_WIDTH[20]))
            images_test.append(images_temp)
        images_test = np.concatenate(images_test)




        label_train = []
        for year in self.train_val_years:
            label_temp = pd.read_feather(op.join(self.path, f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"))
            label_train.append(label_temp)

        label_train = pd.concat(label_train)[[self.target]]

        label_test = []
        for year in self.test_years:
            label_temp = pd.read_feather(op.join(self.path, f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"))
            label_test.append(label_temp)

        label_test = pd.concat(label_test)[[self.target]]



        label_test[self.target] = label_test[self.target].apply(lambda x: 0 if x == 0 else 1)
        label_train[self.target] = label_train[self.target].apply(lambda x: 0 if x == 0 else 1)

        x_train, x_val, y_train, y_val = train_test_split(images_train, label_train, test_size=0.3, random_state=0, shuffle=True)

        return [x_train, x_val, y_train, y_val, images_test, label_test]
   
    def get_dataloader(self, x_train, x_val, y_train, y_val, images_test, label_test):
        
        train_dataset = CNNDataset(x_train, y_train.values)
        val_dataset = CNNDataset(x_val, y_val.values)
        test_dataset = CNNDataset(images_test, label_test.values)

        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return [train_loader, val_loader, test_loader]
    
    def main(self):
        
        print(f"{'Data Processing':-^40}")
        [x_train, x_val, y_train, y_val, images_test, label_test] = self.processing_data()
        
        print(f"{'get_dataloader':-^40}")
        [train_loader, val_loader, test_loader] = self.get_dataloader(x_train, x_val, y_train, y_val, images_test, label_test)
        
        return [train_loader, val_loader, test_loader]
    
class DataLoad2():
    
    def __init__(self, IMAGE_WIDTH, IMAGE_HEIGHT, train_val_years, test_years, target, path, batch_size):
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.train_val_years = train_val_years
        self.test_years = test_years
        self.target = target
        self.path= path
        self.batch_size = batch_size
        
    def processing_data(self):
        images_train = []
        for year in self.train_val_years:
            images_temp = np.memmap(op.join(self.path, f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8, mode='r').reshape(
                                (-1, self.IMAGE_HEIGHT[20], self.IMAGE_WIDTH[20]))
            images_train.append(images_temp)
        images_train = np.concatenate(images_train)

        images_test = []
        for year in self.test_years:
            images_temp = np.memmap(op.join(self.path, f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8, mode='r').reshape(
                                (-1, self.IMAGE_HEIGHT[20], self.IMAGE_WIDTH[20]))
            images_test.append(images_temp)
        images_test = np.concatenate(images_test)




        label_train = []
        for year in self.train_val_years:
            label_temp = pd.read_feather(op.join(self.path, f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"))
            label_train.append(label_temp)

        label_train = pd.concat(label_train)[[self.target]]

        label_test = []
        for year in self.test_years:
            label_temp = pd.read_feather(op.join(self.path, f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"))
            label_test.append(label_temp)

        label_test = pd.concat(label_test)[[self.target]]
        
        images_train = images_train[label_train[self.target].isnull() == False]
        label_train = label_train[label_train[self.target].isnull() == False]
        images_test = images_test[label_test[self.target].isnull() == False]
        label_test = label_test[label_test[self.target].isnull() == False]
        
        x_train, x_val, y_train, y_val = train_test_split(images_train, label_train, test_size=0.3, random_state=0, shuffle=True)

        return [x_train, x_val, y_train, y_val, images_test, label_test]
   
    def get_dataloader(self, x_train, x_val, y_train, y_val, images_test, label_test):
        
        train_dataset = CNNDataset(x_train, y_train.values)
        val_dataset = CNNDataset(x_val, y_val.values)
        test_dataset = CNNDataset(images_test, label_test.values)

        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return [train_loader, val_loader, test_loader]
    
    def main(self):
        
        print(f"{'Data Processing':-^40}")
        [x_train, x_val, y_train, y_val, images_test, label_test] = self.processing_data()
        
        print(f"{'get_dataloader':-^40}")
        [train_loader, val_loader, test_loader] = self.get_dataloader(x_train, x_val, y_train, y_val, images_test, label_test)
        
        return [train_loader, val_loader, test_loader]
    
    
    
    
if __name__ == "__main__":
    
    IMAGE_WIDTH = config.IMAGE_WIDTH
    IMAGE_HEIGHT = config.IMAGE_HEIGHT
    train_val_years = config.train_val_years
    test_years = config.test_years
    target = config.target
    path = config.path
    
    data = DataLoad(IMAGE_WIDTH, IMAGE_HEIGHT, train_val_years, test_years, target, path)
    
    [train_loader, val_loader, test_loader] = data.main()
    
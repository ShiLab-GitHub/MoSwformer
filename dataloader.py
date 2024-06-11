import os
import numpy as np
import torch
from torch.utils.data import Dataset

class data_set(Dataset):
    def __init__(self,data_folder='./BRCA_split/BRCA',
                 train_flag = False
                 ):
        super(data_set, self).__init__()

        if train_flag == True:
            finalname = "_tr.csv"
        else:
            finalname = "_te.csv"

        labels = np.loadtxt(os.path.join(data_folder, "labels"+finalname), delimiter=',')
        self.labels_tensor = torch.LongTensor(labels.astype(int))

        data_1_list = np.loadtxt(os.path.join(data_folder, "1"+finalname), delimiter=',')
        data_2_list = np.loadtxt(os.path.join(data_folder, "2"+finalname), delimiter=',')
        data_3_list = np.loadtxt(os.path.join(data_folder, "3"+finalname), delimiter=',')

        self.data_1_tensor = torch.FloatTensor(data_1_list)
        self.data_2_tensor = torch.FloatTensor(data_2_list)
        self.data_3_tensor = torch.FloatTensor(data_3_list)

        self.data_tensor = torch.cat([self.data_1_tensor, self.data_2_tensor, self.data_3_tensor], dim=1)


    def __len__(self):
        return self.data_1_tensor.shape[0]

    def __getitem__(self, idx):
        sample = dict()

        sample['data1'] = self.data_1_tensor[idx]
        sample['data2'] = self.data_2_tensor[idx]
        sample['data3'] = self.data_3_tensor[idx]

        sample['data'] = self.data_tensor[idx]

        sample['label'] = self.labels_tensor[idx]

        return sample

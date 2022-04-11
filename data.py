import os
import torch
import numpy as np
import pandas as pd
import commons
import random

from torch.utils.data import Dataset, DataLoader, TensorDataset
from transforms import get_transforms

class AudioDataset(Dataset):
    
    def __init__(self, df, path_audio, transform=None, data_dir='data'):
        self.df = df
        self.transform = transform
        self.path_audio = path_audio
        self.data_dir = data_dir
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        
        this_file = self.df.index[idx]
        name = int(this_file)
        path = os.path.join(self.data_dir, str(self.path_audio[name]), str(name)+'.npy')
        
        image = np.load(path)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.Tensor(self.df.values[idx])
    

def parse(a_info):
    ans = []
    total = 0
    
    for k,v in a_info[1]['mood/theme'].items():
        
        total += len(list(v))
        
        ans.append(pd.DataFrame({
            k: 1,
            'file': list(v)
        }))
        
    df = ans[0]
    
    for x in ans[1:]:
        df = pd.merge(df, x, on='file', how='outer')
    df = df.set_index('file').fillna(0)
    df.loc[:, :] = df.loc[:, :].astype('int64')
    return df

def load_info(file, path_audio, data_dir='data'):
    path = os.path.join(data_dir, file)
    with open(path) as f:
        contents= f.read().splitlines()
        for con in contents[1:]:
            con=con.split("\t")[3].split(".")[0].split("/")
            path_audio[int(con[1])]=con[0]
    info = commons.read_file(path)
    df = parse(info)
    return df

def shorten(name, path_audio, seg_size, data_dir = 'data'):
    path = os.path.join(data_dir, str(path_audio[name]), str(name)+'.npy')
    mel = np.load(path)
    mel_len = mel.shape[1]
    offset = random.randint(0, (mel_len - seg_size))
    cut_mel = mel[:, offset:(offset + seg_size)]
    return cut_mel

def load_data(args, data_dir):
    path_audio = {}

    train_df = load_info(args.train, path_audio, data_dir)
    valid_df = load_info(args.valid, path_audio, data_dir)
    test_df = load_info(args.test, path_audio, data_dir)

    test_df = test_df.sort_index()

    train_transform = get_transforms(
        train=True,
        size=args.size,
        wrap_pad_prob=0.5,
        resize_scale=(0.8, 1.0),
        resize_ratio=(1.7, 2.3),
        resize_prob=0.33,
        spec_num_mask=3,
        spec_freq_masking=0.15,
        spec_time_masking=0.20,
        spec_prob=0.5
    )

    train_dataset = AudioDataset(train_df, path_audio, train_transform)

    return train_dataset, valid_df, test_df, path_audio
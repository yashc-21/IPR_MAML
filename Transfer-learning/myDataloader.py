import os
import torch
import pandas as pd
import pickle
import numpy as np

import os
import torch
import pandas as pd
import pickle
import numpy as np


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, path_csv, feature_pkl, fraction, frame_length=120):
        self.df = pd.read_csv(path_csv)
        n_rows = int(fraction * len(self.df))

        self.df = self.df.iloc[:n_rows]

        with open(feature_pkl, "rb") as f:
            self.audio_feat = pickle.load(f)

        self.emotions = {
            "angry": 0,
            "happy": 1,
            "neutral": 2,
            "sad": 3,
        }

        self.df = self.df[ self.df['emotion'].isin(self.emotions.keys()) ]
        self.df =  self.df.reset_index()
        self.num_emotions = len(self.emotions.keys())
        self.frame_length =  frame_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = self.df.loc[idx, 'path']
        emotion = self.df.loc[idx, 'emotion']

        ft = self.audio_feat[ audio_name ]

        if ft.shape[1] != 20: 
            ft = np.transpose( ft, (1, 0) )

        if ft.shape[0] > self.frame_length:
            ft = ft[:self.frame_length, :]
        elif ft.shape[0] < self.frame_length:
            ft = np.block([
                [ft],
                [np.zeros((self.frame_length - ft.shape[0], ft.shape[1]))]
            ])

        ft = torch.from_numpy(ft)

        return ft.float(), self.emotions[ emotion ]


if __name__ == '__main__':
    ds = DataLoader(path_csvs=['../data/emovo/emovo_train.csv'], feature_pkls=['../data/emovo/emovo.pkl'])
    x, y = ds.__getitem__(5)
    print (x.shape)
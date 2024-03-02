import torch
from torch.utils.data import Dataset
import re

import sys
sys.path.append('../')
from onehot_encode import *


class ScreenDataset(Dataset):
    def __init__(self, data, tokenizer, use_property=False):
        self.use_property = use_property
        self.length = data.shape[0]
        if self.use_property:
            self.feature = data[['Mw', 'charge of all', 'positive_charge', 'negative_charge',
                                 'polar_number', 'unpolar_number', 'ph_number', 'hydrophobicity',
                                 'vdW_volume']]
        seq = []
        for i in data['seq'].values.tolist():
            data_list = re.findall(".{1}", i)
            seq.append(" ".join(data_list))

        self.all_input = tokenizer.batch_encode_plus(seq,padding=True)
        # all_input_ids = tokenizer(seq, add_special_tokens=True, truncation=True, max_length=30)
        # all_input_ids = all_input_ids["input_ids"]
        # self.dict_ids = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in all_input_ids]


    def __getitem__(self, idx):
        # example = dict(self.dict_ids[idx])
        example = {'input_ids':torch.tensor(self.all_input[idx].ids),
                   'attention_mask':torch.tensor(self.all_input[idx].attention_mask)}
        if self.use_property:
            feature = torch.from_numpy(self.feature.iloc[idx].values)
            example.update({'feature': feature})
        return example

    def __len__(self):
        return self.length


class lstm_dataset(torch.utils.data.Dataset):
    def __init__(self, data,
                 encoder=DirectSequenceEncoder(ResidueIdentityEncoder(RESIDUES, place_holder='X')),
                 use_property=True):
        self.seq = data['seq'].values
        self.length = data.shape[0]
        self.labels = [0 for _ in range(self.length)]
        self.use_property = use_property
        if self.use_property:
            self.feature = data[['Mw', 'charge of all', 'positive_charge', 'negative_charge',
                                 'polar_number', 'unpolar_number', 'ph_number', 'hydrophobicity',
                                 'vdW_volume']]
        self._encoder = encoder

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        length = len(self.seq[idx])
        seq = self._encoder(self.seq[idx]).astype('uint8')
        label = self.labels[idx]
        if self.use_property:
            feature = torch.from_numpy(self.feature.iloc[idx].values)
        else:
            feature = None
        return torch.tensor(seq), torch.tensor(label), feature, length


class cnn_dataset(torch.utils.data.Dataset):
    def __init__(self, data,
                 encoder=DirectSequenceEncoder(ResidueIdentityEncoder(RESIDUES, place_holder='X')),
                 use_property=True):
        self.seq = data['seq'].values
        self.length = data.shape[0]
        self.labels = [0 for _ in range(self.length)]
        self.use_property = use_property
        if self.use_property:
            self.feature = data[['Mw', 'charge of all', 'positive_charge', 'negative_charge',
                                 'polar_number', 'unpolar_number', 'ph_number', 'hydrophobicity',
                                 'vdW_volume']]
        self._encoder = encoder

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        length = len(self.seq[idx])
        seq = self._encoder(self.seq[idx]).astype('uint8')
        label = self.labels[idx]
        if self.use_property:
            feature = torch.from_numpy(self.feature.iloc[idx].values)
        else:
            feature = None
        return torch.tensor(seq), torch.tensor(label), feature, length
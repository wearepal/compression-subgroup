import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


# Defined classes
class Subset(Dataset):
    
    """
    Subsets a dataset while preserving original indexing.

    NOTE: torch.utils.dataset.Subset loses original indexing.
    """
    
    def __init__(self, dataset, indices):
        
        self.dataset = dataset
        self.indices = indices

        self.group_array = self.get_group_array(re_evaluate=True)
        self.label_array = self.get_label_array(re_evaluate=True)

    def __len__(self):
        
        return len(self.indices)
        
    def __getitem__(self, idx):
        
        return self.dataset[self.indices[idx]]

    def get_group_array(self, re_evaluate=True):
        
        """Return an array [g_x1, g_x2, ...]"""
        # setting re_evaluate=False helps us over-write the group array if necessary (2-group DRO)
        
        if re_evaluate:
            group_array = self.dataset.get_group_array()[self.indices]        
            assert len(group_array) == len(self)
            return group_array
        
        else:
            return self.group_array

    def get_label_array(self, re_evaluate=True):
        
        if re_evaluate:
            label_array = self.dataset.get_label_array()[self.indices]
            assert len(label_array) == len(self)
            return label_array
        
        else:
            return self.label_array


class ConfounderDataset(Dataset):

    def __len__(self):
        
        return len(self.filename_array)

    def __getitem__(self, idx):
        
        y = self.y_array[idx]
        g = self.group_array[idx]

        img_filename = os.path.join(self.root_dir, self.filename_array[idx])
        x = Image.open(img_filename).convert('RGB')
        
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            x = self.train_transform(x)
        
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and self.eval_transform):
            x = self.eval_transform(x)
        
        return x, y, g, idx

    def get_group_array(self):
        
        return self.group_array

    def get_label_array(self):
        
        return self.y_array

    def get_splits(self, splits, train_frac=1.0):
        
        subsets = {}
        for split in splits:
            
            assert split in ('train', 'val', 'test'), f'{split} is not a valid split'
            mask = self.split_array == self.split_dict[split]

            indices = np.where(mask)[0]
            
            if train_frac < 1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            
            subsets[split] = Subset(self, indices)
        
        return subsets
        
    def group_str(self, group_idx):
        
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        
        return group_name

import os
import numpy as np
import pandas as pd

from confounder_dataset import ConfounderDataset


# Defined classes
class JigsawDataset(ConfounderDataset):
    
    """
    Jigsaw dataset. We only consider the subset of examples with identity annotations.
    Labels are 1 if target_name > 0.5, and 0 otherwise.

    95% of tokens have max_length <= 220, and 99.9% have max_length <= 300
    """

    def __init__(self, root_dir, target_name, confounder_names, tokenizer):

        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.max_length = 300
        
        # Read in metadata
        self.metadata_df = pd.read_csv(os.path.join(self.root_dir, 'all_data_with_identities.csv'), index_col=0)

        # Get the y values
        self.y_array = (self.metadata_df[self.target_name].values >= 0.5).astype('long')
        self.n_classes = len(np.unique(self.y_array))

        if self.confounder_names[0] == 'only_label':
            self.n_groups = self.n_classes
            self.group_array = self.y_array
        
        else:
            
            # Confounders are all binary
            # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
            self.n_confounders = len(self.confounder_names)
            confounders = (self.metadata_df.loc[:, self.confounder_names] >= 0.5).values
            self.confounder_array = confounders @ np.power(2, np.arange(self.n_confounders))

            # Map to groups
            self.n_groups = self.n_classes * pow(2, self.n_confounders)
            self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype("int")

        # Extract splits
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}
        for split in self.split_dict:
            self.metadata_df.loc[self.metadata_df['split'] == split, 'split'] = self.split_dict[split]

        self.split_array = self.metadata_df['split'].values

        # Extract text
        self.text_array = list(self.metadata_df['comment_text'])
        self.tokenizer = tokenizer

    def __len__(self):
        
        return len(self.y_array)

    def __getitem__(self, idx):
        
        text = self.text_array[idx]
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        try:
            sample = {
                'input_ids': tokens['input_ids'].flatten(), 
                'attention_mask': tokens['attention_mask'].flatten(), 
                'token_type_ids': tokens['token_type_ids'].flatten(), 
                'labels': self.y_array[idx]
            }
        
        except:
            sample = {
                'input_ids': tokens['input_ids'].flatten(), 
                'attention_mask': tokens['attention_mask'].flatten(), 
                'labels': self.y_array[idx]
            }
        
        return sample

    def get_label_array(self):
        
        return self.y_array

    def get_group_array(self):
        
        return self.group_array

    def group_str(self, group_idx):
        
        if self.n_groups == self.n_classes:
            y = group_idx
            group_name = f'{self.target_name} = {int(y)}'
        
        else:
            y = group_idx // (self.n_groups / self.n_classes)
            c = group_idx % (self.n_groups // self.n_classes)

            group_name = f'{self.target_name} = {int(y)}'
            bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
            for attr_idx, attr_name in enumerate(self.confounder_names):
                group_name += f', {attr_name} = {bin_str[attr_idx]}'
        
        return group_name

import numpy as np
import pandas as pd

from datasets import load_dataset

from confounder_dataset import ConfounderDataset


# Defined classes
class ScotusDataset(ConfounderDataset):
    
    def __init__(self, root_dir, target_name, confounder_names, tokenizer):
        
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.max_length = 512

        # Read in metadata
        data = load_dataset('coastalcph/fairlex', 'scotus')
        self.metadata_df = pd.concat([data['train'].to_pandas(), data['validation'].to_pandas(), data['test'].to_pandas()], ignore_index=True)

        # Get the y values
        self.y_array = self.metadata_df[self.target_name].values
        self.n_classes = len(np.unique(self.y_array))
        
        assert len(self.confounder_names) == 1
        self.confounder_array = self.metadata_df[self.confounder_names].values.flatten()
        self.n_confounders = len(self.confounder_names)

        # Map to groups
        self.n_groups = len(np.unique(self.confounder_array)) * self.n_classes
        self.group_array = (self.y_array * (self.n_groups / self.n_classes) + self.confounder_array).astype("int")

        # Extract splits
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}
        self.split_array = np.array(
            [0] * len(data['train']) + [1] * len(data['validation']) + [2] * len(data['test'])
        )
        self.metadata_df['split'] = self.split_array

        # Extract text
        self.text_array = data['train']['text'] + data['validation']['text'] + data['test']['text']
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
            return_tensors='pt',
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

    def group_str(self, group_idx):
        
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        attr_name = self.confounder_names
        group_name = f'{self.target_name} = {int(y)}, {attr_name} = {int(c)}'
        return group_name

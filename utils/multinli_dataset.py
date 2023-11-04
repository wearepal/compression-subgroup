import os
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer

from confounder_dataset import ConfounderDataset


# Defined classes
class MultiNLIDataset(ConfounderDataset):
    
    """
    MultiNLI dataset.
    label_dict = {
        'contradiction': 0,
        'entailment': 1,
        'neutral': 2
    }
    # Negation words taken from https://arxiv.org/pdf/1803.02324.pdf
    negation_words = ['nobody', 'no', 'never', 'nothing']
    """
    
    def __init__(self, root_dir, target_name, confounder_names, tokenizer, labels):
        
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.max_length = 128
        
        assert target_name in ['gold_label_random', 'gold_label_preset']

        # Read in metadata
        type_of_split = target_name.split('_')[-1]
        self.metadata_df = pd.read_csv(os.path.join(self.root_dir, f'metadata_{type_of_split}.csv'), index_col=0)

        # Get the y values
        # gold_label is hardcoded
        self.y_array = self.metadata_df['gold_label'][self.metadata_df['gold_label'].isin(labels)].values
        if len(labels) == 2:
            self.y_array[self.y_array == labels[0]] = 0
            self.y_array[self.y_array == labels[1]] = 1
        self.n_classes = len(np.unique(self.y_array))
        
        assert len(self.confounder_names) == 1
        self.confounder_array = self.metadata_df[self.confounder_names[0]][self.metadata_df['gold_label'].isin(labels)].values
        self.n_confounders = len(self.confounder_names)

        # Map to groups
        self.n_groups = len(np.unique(self.confounder_array)) * self.n_classes
        self.group_array = (self.y_array * (self.n_groups / self.n_classes) + self.confounder_array).astype("int")

        # Extract splits
        self.split_array = self.metadata_df['split'][self.metadata_df['gold_label'].isin(labels)].values
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        # Load features
        self.features_array = []
        for feature_file in [
            'cached_train_bert-base-uncased_128_mnli', 
            'cached_dev_bert-base-uncased_128_mnli', 
            'cached_dev_bert-base-uncased_128_mnli-mm'
        ]:

            features = torch.load(os.path.join(self.root_dir, feature_file))
            self.features_array += features

        # Extract features
        all_input_ids = []
        all_input_masks = []
        all_segment_ids = []
        all_label_ids = []

        for f in self.features_array:
            if f.label_id in labels:
                all_input_ids.append(f.input_ids)
                all_input_masks.append(f.input_mask)
                all_segment_ids.append(f.segment_ids)
                all_label_ids.append(f.label_id)

        self.all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        self.all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
        self.all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        self.all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)

        if len(labels) == 2:
            self.all_label_ids[self.all_label_ids == labels[0]] = 0
            self.all_label_ids[self.all_label_ids == labels[1]] = 1

        self.x_array = torch.stack((self.all_input_ids, self.all_input_masks, self.all_segment_ids), dim=2)
        assert np.all(np.array(self.all_label_ids) == self.y_array)

        # Extract text
        self.decoder = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_array = self.decoder.batch_decode(self.all_input_ids)
        self.tokenizer = tokenizer

    def __len__(self):
        
        return len(self.y_array)

    def __getitem__(self, idx):
        
        text = self.text_array[idx]
        tokens = self.tokenizer(
            text.partition('[CLS]')[2].partition('[SEP]')[0].strip(),
            text.partition('[SEP]')[2].partition('[SEP]')[0].strip(),
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

    def group_str(self, group_idx):
        
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        attr_name = self.confounder_names
        group_name = f'{self.target_name} = {int(y)}, {attr_name} = {int(c)}'
        return group_name

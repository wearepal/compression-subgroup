import torch

from torch.utils.data import Dataset


# Defined functions
def prepare_confounder_data(path, dataset, tokenizer=None):

    if 'multinli' in dataset:
        
        from multinli_dataset import MultiNLIDataset
        full_data = MultiNLIDataset(
            root_dir=path,
            target_name='gold_label_random', # 'gold_label_random', 'gold_label_preset'
            confounder_names=['sentence2_has_negation'],
            tokenizer=tokenizer,
            labels=[0, 1, 2] if dataset == 'multinli' else [int(dataset.split('_')[1]), int(dataset.split('_')[2])]
        )
    
    elif dataset == 'jigsaw':
        
        from jigsaw_dataset import JigsawDataset
        full_data = JigsawDataset(
            root_dir=path,
            target_name='toxicity',
            confounder_names=['identity_any'],
            tokenizer=tokenizer
        )

    elif dataset == 'scotus':

        from scotus_dataset import SCOTUSDataset
        full_data = SCOTUSDataset(
            root_dir=path,
            target_name='label',
            confounder_names=['decision_direction'],
            tokenizer=tokenizer
        )
    
    splits = ['train', 'val', 'test']
    subsets = full_data.get_splits(splits, train_frac=1.0)
    
    data = {}
    for split in splits:
        
        data[f'{split}_data'] = DRODataset(
            subsets[split],
            process_item_fn=None,
            n_groups=full_data.n_groups,
            n_classes=full_data.n_classes,
            group_str_fn=full_data.group_str
        )
    
    return data


# Defined classes
class DRODataset(Dataset):
    
    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn):
        
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn

        group_array = self.get_group_array()
        y_array = self.get_label_array()

        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        
        self._group_counts = ((torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float())
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()

    def __len__(self):
        
        return len(self.dataset)

    def __getitem__(self, idx):
        
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def input_size(self):
        
        for x, y, g, _ in self:
            return x.size()

    def get_label_array(self):
        
        if self.process_item is None:
            return self.dataset.get_label_array()
        else:
            raise NotImplementedError

    def get_group_array(self):
        
        if self.process_item is None:
            return self.dataset.get_group_array()
        else:
            raise NotImplementedError

    def class_counts(self):
        
        return self._y_counts

    def group_counts(self):
        
        return self._group_counts

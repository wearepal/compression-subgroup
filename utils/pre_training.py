import json
import os
import torch

from torch.utils.data import Dataset
from transformers import Trainer, EarlyStoppingCallback


# Defined functions
def pre_train_model(args, model, train_data, val_data):

  trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
  )
  trainer.train()

  # Save the model
  model.save_pretrained(args.output_dir)

  # Save the loss
  with open(os.path.join(args.output_dir, 'loss.txt'), 'w') as file:
    for obj in trainer.state.log_history:
      file.write(json.dumps(obj))
      file.write('\n\n')


# Defined classes
class MLMDataset(Dataset):

  def __init__(self, tokenizer, data, seed):
    self.tokenizer = tokenizer
    self.data = data
    self.seed = seed

    vocab = self.tokenizer.get_vocab()
    special_tokens = self.tokenizer.special_tokens_map

    self.special_ids = [vocab[special_token] for special_token in special_tokens.values()]
    self.mask_id = vocab[special_tokens['mask_token']]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    text = self.data[idx]

    # Get the input ids and labels
    input_ids, labels = self.__get_inputs(text)

    # Mask the input ids
    input_ids = self.__get_mask(input_ids)

    return {'input_ids': input_ids, 'labels': labels}

  def __get_inputs(self, text):
    if ('[CLS]' in text and '[SEP]' in text):
      tokens = self.tokenizer(
        text.partition('[CLS]')[2].partition('[SEP]')[0].strip(),
        text.partition('[SEP]')[2].partition('[SEP]')[0].strip(),
        padding='max_length', 
        truncation=True, 
        return_tensors='pt',
      )
    else:      
      tokens = self.tokenizer(
        text=text, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
      )
    
    input_ids = tokens['input_ids'].flatten()
    labels = input_ids.clone()
    return input_ids, labels

  def __get_mask(self, input_ids):

    # Create a random array for masking using a given seed
    torch.manual_seed(self.seed)
    rand = torch.rand(len(input_ids))

    # Set to True the indices whose value is less than 0.15 and whose input id is not a special token
    mask_arr = (rand < 0.15) * ~sum(input_ids == i for i in self.special_ids).bool()

    # Mask the input ids using the masking array
    selection = mask_arr.nonzero().flatten().tolist()
    input_ids[selection] = self.mask_id

    return input_ids

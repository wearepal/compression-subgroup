import argparse
import itertools
import os
import pandas as pd
import shutil
import torch

import sys
sys.path.append(os.path.join('utils'))

from copy import deepcopy
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments

from utils.confounder_utils import prepare_confounder_data

from utils.pre_training import pre_train_model
from utils.pre_training import MLMDataset
from utils.classification import train_model, test_model

from utils.pruning import prune_model
from utils.quantization import BertForSequenceClassification
from utils.vocabulary import vocab_transfer


def main():

  # Define the arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--data', 
    type=str, 
    choices=['multinli', 'jigsaw', 'scotus', 'multinli_0_1', 'multinli_0_2', 'multinli_1_2'], 
    required=True, 
    help='The dataset to use.'
  )
  parser.add_argument(
    '--kd', 
    type=str, 
    choices=[
      'bert_medium', 'bert_small', 'bert_mini', 'bert_tiny', 'distilbert', 'tinybert_6', 'tinybert_4', 
      'tinybert_ahe', 'tinybert_ah', 'tinybert_ae', 'tinybert_he', 'tinybert_a', 'tinybert_h', 'tinybert_e'
    ],
    required=False, 
    help='The distilled model to use.'
  )
  parser.add_argument(
    '--up', 
    type=int, 
    choices=[20, 40, 60, 80], 
    required=False, 
    help='The unstructured pruning amount to use.'
  )
  parser.add_argument(
    '--sp', 
    type=int, 
    choices=[20, 40, 60, 80], 
    required=False, 
    help='The structured pruning amount to use.'
  )
  parser.add_argument(
    '--qn', 
    type=str, 
    choices=['dq', 'sq', 'qat'], 
    required=False, 
    help='The quantization method to use.'
  )
  parser.add_argument(
    '--vt', 
    type=int, 
    choices=[100, 75, 50, 25], 
    required=False, 
    help='The vocabulary transfer size to use.'
  )

  args = parser.parse_args()

  # Check the arguments
  if {args.kd, args.up, args.sp, args.qn, args.vt} == {None}:
    pass

  elif args.kd is not None and {args.up, args.sp, args.qn, args.vt} == {None}:
    pass

  elif args.up is not None and {args.kd, args.sp, args.qn, args.vt} == {None}:
    pass

  elif args.sp is not None and {args.kd, args.up, args.qn, args.vt} == {None}:
    pass

  elif args.qn is not None and {args.kd, args.up, args.sp, args.vt} == {None}:
    pass

  elif args.vt is not None and {args.kd, args.up, args.sp, args.qn} == {None}:
    pass

  else:
    raise NotImplementedError
    
  # Set the hyperparameters
  DATA = args.data
  KD = args.kd
  UP = args.up
  SP = args.sp
  QN = args.qn
  VT = args.vt


  # Define the variables
  if KD is None:
    model_path = 'bert-base-uncased'

  elif KD in ['bert_medium', 'bert_small', 'bert_mini', 'bert_tiny']:
    model_path = f"prajjwal1/{KD.replace('_', '-')}"

  elif KD == 'distilbert':
    model_path = 'distilbert-base-uncased'

  elif KD == 'tinybert_6':
    model_path = 'huawei-noah/TinyBERT_General_6L_768D'

  elif KD == 'tinybert_4':
    model_path = 'huawei-noah/TinyBERT_General_4L_312D'

  elif KD in [
    'tinybert_ahe', 'tinybert_ah', 'tinybert_ae', 'tinybert_he', 'tinybert_a', 'tinybert_h', 'tinybert_e'
  ]:
    model_path = os.path.join('models', 'tinybert_6', KD.split('_')[1])

  batch_size = 32 if 'multinli' in DATA else 16

  # Define the save path
  if {KD, UP, SP, QN, VT} == {None}:
    save_path = os.path.join('logs', DATA, 'base')

  elif KD is not None:
    save_path = os.path.join('logs', DATA, 'kd', KD)

  elif UP is not None:
    save_path = os.path.join('logs', DATA, 'up', str(UP))

  elif SP is not None:
    save_path = os.path.join('logs', DATA, 'sp', str(SP))

  elif QN is not None:
    save_path = os.path.join('logs', DATA, 'qn', QN)

  elif VT is not None:
    save_path = os.path.join('logs', DATA, 'vt', str(VT))

  # Define the arguments
  args = TrainingArguments(
    output_dir='',
    evaluation_strategy='epoch',
    learning_rate=0.00002 if 'multinli' in DATA else 0.00001,
    weight_decay=0.01 if DATA == 'jigsaw' else 0,
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True
  )


  # Run the experiments
  for i in range(1, 6):

    # Set the seed
    set_seed(i)


    # Load the tokenizer
    if {KD, UP, SP, QN, VT} == {None} or UP is not None or SP is not None or QN is not None:
      tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    elif KD is not None:
      tokenizer = AutoTokenizer.from_pretrained(
        model_path if (KD == 'distilbert' or 'tinybert_6' in KD or 'tinybert_4' in KD) else 'bert-base-uncased'
      )

    elif VT is not None:
      tokenizer = AutoTokenizer.from_pretrained(os.path.join('tokenizers', DATA, str(VT)))

    # Load the dataset
    dataset = prepare_confounder_data(os.path.join('data', DATA.split('_')[0]), DATA, tokenizer)
    num_labels = dataset['train_data'].n_classes


    # Pre-train the model
    if VT is not None:
      mlm_path=os.path.join(save_path, f'seed_{i}', 'masked_lm')
      
      mlm_args=deepcopy(args)
      mlm_args.output_dir=mlm_path
      mlm_args.per_device_train_batch_size=8
      mlm_args.per_device_eval_batch_size=8
      mlm_args.num_train_epochs=1

      masked_lm = AutoModelForMaskedLM.from_pretrained(model_path)
      vocab_transfer(AutoTokenizer.from_pretrained('bert-base-uncased'), tokenizer, masked_lm, 'FVT')

      text = pd.Series(dataset['train_data'].dataset.dataset.text_array)
      X_train = text[dataset['train_data'].dataset.indices].reset_index(drop=True)
      X_val = text[dataset['val_data'].dataset.indices].reset_index(drop=True)

      pre_train_model(mlm_args, masked_lm, MLMDataset(tokenizer, X_train, seed=i), MLMDataset(tokenizer, X_val, seed=i))


    # Train the model
    train_path=os.path.join(save_path, f'seed_{i}')
  
    train_args=deepcopy(args)
    train_args.output_dir=train_path
    train_args.per_device_train_batch_size=batch_size
    train_args.per_device_eval_batch_size=batch_size
    train_args.num_train_epochs=20 if DATA == 'scotus' else 5
    train_args.metric_for_best_model='average'
    train_args.greater_is_better=True
    
    if {KD, UP, SP, QN, VT} == {None} or KD is not None or VT is not None:
      model = AutoModelForSequenceClassification.from_pretrained(model_path if VT is None else mlm_path, num_labels=num_labels)
      train_model(train_args, model, dataset['train_data'], dataset['val_data'])

    elif QN == 'sq':
      model = BertForSequenceClassification.from_pretrained(os.path.join('logs', DATA, 'base', f'seed_{i}'), num_labels=num_labels)

      qconfig_spec = dict(zip(
        {torch.nn.Linear, torch.quantization.stubs.QuantStub, torch.quantization.stubs.DeQuantStub}, 
        itertools.repeat(torch.ao.quantization.get_default_qconfig('x86'))
      ))
      torch.ao.quantization.propagate_qconfig_(model, qconfig_spec)
      torch.ao.quantization.prepare(model, inplace=True)

    elif QN == 'qat':
      model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
      
      qconfig_spec = dict(zip(
        {torch.nn.Linear, torch.ao.quantization.QuantStub, torch.ao.quantization.DeQuantStub}, 
        itertools.repeat(torch.ao.quantization.get_default_qat_qconfig('x86'))
      ))
      torch.ao.quantization.propagate_qconfig_(model, qconfig_spec)
      torch.ao.quantization.prepare_qat(model.train(), inplace=True)

      train_model(train_args, model, dataset['train_data'], dataset['val_data'])
      os.remove(os.path.join(train_path, 'pytorch_model.bin'))
      os.remove(os.path.join(train_path, 'config.json'))

    else:
      model = AutoModelForSequenceClassification.from_pretrained(os.path.join('logs', DATA, 'base', f'seed_{i}'), num_labels=num_labels)


    # Post-process the model
    if UP is not None or SP is not None:
      prune_model(
        train_args, 
        model, 
        dataset['train_data'], 
        dataset['val_data'], 
        UP / 100 if UP is not None else SP / 100, 
        'unstructured' if UP is not None else 'structured'
      )
    
    elif QN == 'dq':
      torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, inplace=True)

    elif QN == 'sq':
      test_model(train_args, model, dataset['train_data'])
      os.remove(os.path.join(train_path, 'predictions.csv'))
      os.remove(os.path.join(train_path, 'results.txt'))

    if QN in ['sq', 'qat']:
      model = model.to('cpu')
      torch.ao.quantization.convert(model, inplace=True)

    # Clean up the pre-trained model
    if VT is not None:
      shutil.rmtree(mlm_path)


    # Evaluate the model
    test_model(
      TrainingArguments( 
        output_dir=train_path,
        remove_unused_columns=False,
        per_device_eval_batch_size=batch_size,
        no_cuda=True if QN is not None else False
      ),
      model,
      dataset['test_data']
    )


if __name__ == '__main__':
    main()

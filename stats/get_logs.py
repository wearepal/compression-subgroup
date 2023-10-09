import os
import pandas as pd

import wandb
wandb.login()

os.environ['WANDB_SILENT'] = 'true'


def main():

    # Define the configurations
    configs = [
        'base', 

        'kd@bert_medium', 'kd@bert_small', 'kd@bert_mini', 'kd@bert_tiny', 'kd@distilbert', 'kd@tinybert_6', 'kd@tinybert_4', 

        'kd@tinybert_ahe', 'kd@tinybert_ah', 'kd@tinybert_ae', 'kd@tinybert_he', 'kd@tinybert_a', 'kd@tinybert_h', 'kd@tinybert_e',

        'sp@20', 'sp@40', 'sp@60', 'sp@80', 

        'qn@dq', 'qn@sq', 'qn@qat', 

        'vt@100', 'vt@75', 'vt@50', 'vt@25',

        'up@20', 'up@40', 'up@60', 'up@80'
    ]

    for data in ['multinli', 'jigsaw', 'scotus', 'multinli_0_1', 'multinli_0_2', 'multinli_1_2']:

        for config in configs:

            folder = config if config == 'base' else os.path.join(config.split('@')[0], config.split('@')[1])
            
            for i in range(1, 6):

                # Initialize the wandb run
                wandb.init(project='themis-enas', reinit=True, group=data, job_type=folder, name=f'seed_{i}')
                
                # Load the results
                y = pd.read_csv(
                    os.path.join('..', 'logs', data, folder, f'seed_{i}', 'results.txt'),
                    delimiter='\n',
                    names=['Text'],
                    header=None
                )

                # Filter the results
                y_average = y[y['Text'].str.contains('Average acc: ')].reset_index(drop=True)
                y_worst = y[y['Text'].str.contains('Worst acc: ')].reset_index(drop=True)
                y_size = y[y['Text'].str.contains('Model size: ')].reset_index(drop=True)
                y_acc = y[y['Text'].str.contains('acc = ')].reset_index(drop=True)

                # Get the values
                wandb.log({
                    'test': {
                        'average': y_average['Text'].str.split('Average acc: ').str[1].str.split(' ').str[0].astype(float)[0], 
                        'worst': y_worst['Text'].str.split('Worst acc: ').str[1].str.split(' ').str[0].astype(float)[0],
                        'size': y_size['Text'].str.split('Model size: ').str[1].str.split(' ').str[0].astype(float)[0]
                    }
                })

                acc = y_acc['Text'].str.split('acc = ').str[1].str.split(' ').str[0].astype(float)
                for group_idx in range(len(acc)):
                    wandb.log({'test': {f'subgroup_{group_idx}': acc[group_idx]}})

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()

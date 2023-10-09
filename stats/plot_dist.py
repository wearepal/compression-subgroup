import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

import os
import pandas as pd


def main():

    # Define the configurations
    settings = {
        'kd': ['base', 'bert_medium', 'bert_small', 'bert_mini', 'bert_tiny', 'distilbert', 'tinybert_6', 'tinybert_4']
    }

    labels = {
        'base': '$\mathcal{BERT}_{Base}$',

        'kd_bert_medium': '$\mathcal{BERT}_{Medium}$', 'kd_bert_small': '$\mathcal{BERT}_{Small}$', 
        'kd_bert_mini': '$\mathcal{BERT}_{Mini}$', 'kd_bert_tiny': '$\mathcal{BERT}_{Tiny}$',

        'kd_distilbert': '$\mathcal{Distilbert}$',

        'kd_tinybert_6': '$\mathcal{TinyBERT}_{6}$', 'kd_tinybert_4': '$\mathcal{TinyBERT}_{4}$'
    }
 
    fig = plt.figure(figsize=(25, 5))

    for i, data in enumerate(['multinli', 'jigsaw']):

        # Create the dataframe
        df = pd.DataFrame()

        for method, configs in settings.items():

            for config in configs:
            
                acc = pd.DataFrame()
                for j in range(1, 6):
                    
                    # Load the results
                    y = pd.read_csv(
                        os.path.join(
                            '..', 'logs', data, config if config == 'base' else os.path.join(method, config), f'seed_{j}', 'results.txt'
                        ),
                        delimiter='\n',
                        names=['Text'],
                        header=None
                    )

                    # Filter the results
                    y_acc = y[y['Text'].str.contains('acc = ')].reset_index(drop=True)

                    # Get the accuracies
                    acc = acc.append(y_acc['Text'].str.split('acc = ').str[1].str.split(' ').str[0].astype(float), ignore_index=True)

                # Append the accuracies
                df[labels[config if config == 'base' else f'{method}_{config}']] = acc.mean(axis=0)

        # Change the index
        index = {'multinli': [57498, 11158, 67376, 1521, 66630, 1992], 'jigsaw': [148186, 90337, 12731, 17784]}
        df.index = [f'{k} [{index[data][k]}]' for k in range(len(index[data]))]

        # Plot the dataframe
        ax = plt.subplot(1, 2, 1 + i)
        
        df.plot.bar(
            ax=ax, title='MultiNLI' if data == 'multinli' else 'CivilComments', xlabel='Subgroup', ylabel='Accuracy', rot=0, alpha=0.8, legend=False
        )
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(df.columns))
    
    # Save the plot
    if not os.path.isdir(os.path.join('..', 'figures')):
        os.makedirs(os.path.join('..', 'figures'))
    plt.savefig(os.path.join('..', 'figures', 'subgroups.pdf'), bbox_inches='tight')
    print(os.path.exists(os.path.join('..', 'figures', 'subgroups.pdf')))


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import numpy as np
import os
import pandas as pd
import seaborn as sns


def main():

    # Define the configurations
    configs = {
        'base': None,

        'bert_medium': 'kd', 'bert_small': 'kd', 'bert_mini': 'kd', 'bert_tiny': 'kd',

        'distilbert': 'kd',
        
        'tinybert_6': 'kd', 'tinybert_4': 'kd',

        '20': 'sp', '40': 'sp', '60': 'sp', '80': 'sp',

        'dq': 'qn', 'sq': 'qn', 'qat': 'qn',
        
        '100': 'vt', '75': 'vt', '50': 'vt', '25': 'vt'
    }

    fig = plt.figure(figsize=(30, 5))

    for i, data in enumerate(['multinli_0_1', 'multinli_0_2', 'multinli_1_2']):

        # Create the dataframe
        df = pd.DataFrame()

        average_all = []
        worst_all = []
        size_all = []
        for config in configs.keys():
            
            average = []
            worst = []
            size = []
            for j in range(1, 6):
                
                # Load the results
                y = pd.read_csv(
                    os.path.join(
                        '..', 'logs', data, config if config == 'base' else os.path.join(configs[config], config), f'seed_{j}', 'results.txt'
                    ),
                    delimiter='\n',
                    names=['Text'],
                    header=None
                )

                # Filter the results
                y_average = y[y['Text'].str.contains('Average acc: ')].reset_index(drop=True)
                y_worst = y[y['Text'].str.contains('Worst acc: ')].reset_index(drop=True)
                y_size = y[y['Text'].str.contains('Model size: ')].reset_index(drop=True)

                # Get the values
                average.append(y_average['Text'].str.split('Average acc: ').str[1].str.split(' ').str[0].astype(float))
                worst.append(y_worst['Text'].str.split('Worst acc: ').str[1].str.split(' ').str[0].astype(float))
                size.append(y_size['Text'].str.split('Model size: ').str[1].str.split(' ').str[0].astype(float))

            # Append the means
            average_all.append(np.mean(average))
            worst_all.append(np.mean(worst))
            size_all.append(np.mean(size))

        # Add the means to the dataframe
        df['Average'] = average_all
        df['Worst'] = worst_all
        df['Size'] = size_all

        # Change the index
        df.index = [
            '$\mathcal{BERT}_{Base}$',

            '$\mathcal{BERT}_{Medium}$', '$\mathcal{BERT}_{Small}$', '$\mathcal{BERT}_{Mini}$', '$\mathcal{BERT}_{Tiny}$',

            '$\mathcal{Distilbert}$',

            '$\mathcal{TinyBERT}_{6}$', '$\mathcal{TinyBERT}_{4}$',

            '$\mathcal{BERT}_{PR20}$', '$\mathcal{BERT}_{PR40}$', '$\mathcal{BERT}_{PR60}$', '$\mathcal{BERT}_{PR80}$',

            '$\mathcal{BERT}_{DQ}$', '$\mathcal{BERT}_{SQ}$', '$\mathcal{BERT}_{QAT}$',

            '$\mathcal{BERT}_{VT100}$', '$\mathcal{BERT}_{VT75}$', '$\mathcal{BERT}_{VT50}$', '$\mathcal{BERT}_{VT25}$'
        ]

        # Plot the dataframe
        ax = plt.subplot(1, 3, 1 + i)

        markers = {
            '$\mathcal{BERT}_{Base}$': 'o',

            '$\mathcal{BERT}_{Medium}$': 'v', '$\mathcal{BERT}_{Small}$': 'v', '$\mathcal{BERT}_{Mini}$': 'v', '$\mathcal{BERT}_{Tiny}$': 'v',

            '$\mathcal{Distilbert}$': 'p',
            
            '$\mathcal{TinyBERT}_{6}$': 'P', '$\mathcal{TinyBERT}_{4}$': 'P',

            '$\mathcal{BERT}_{PR20}$': '*', '$\mathcal{BERT}_{PR40}$': '*', '$\mathcal{BERT}_{PR60}$': '*', '$\mathcal{BERT}_{PR80}$': '*',

            '$\mathcal{BERT}_{DQ}$': 'H', '$\mathcal{BERT}_{SQ}$': 'H', '$\mathcal{BERT}_{QAT}$': 'H',
            
            '$\mathcal{BERT}_{VT100}$': 'X', '$\mathcal{BERT}_{VT75}$': 'X', '$\mathcal{BERT}_{VT50}$': 'X', '$\mathcal{BERT}_{VT25}$': 'X'
        }

        cmap = sns.blend_palette(
            ['red', 'orange', 'yellow', 'lime', 'green', 'blue', 'cyan', 'pink', 'purple', 'peachpuff', 'brown', 'grey', 'black'], len(df.index)
        )
        sns.set_palette(cmap, n_colors=len(df.index))
        colors = sns.color_palette(n_colors=len(df.index))

        for i, config in enumerate(df.index):
            plt.scatter(
                x=df.loc[config]['Average'], 
                y=df.loc[config]['Worst'], 
                s=df.loc[config]['Size'], 
                color=colors[i], 
                marker=markers[config], 
                label=config
            ) 
        
        if data == 'multinli_0_1':
            plt.title('MultiNLI (0, 1)')
        
        elif data == 'multinli_0_2':
            plt.title('MultiNLI (0, 2)')

        elif data == 'multinli_1_2':
            plt.title('MultiNLI (1, 2)')

        plt.xlabel('Average Accuracy')
        plt.ylabel('Worst-group Accuracy')

    handles, labels = ax.get_legend_handles_labels()
    for handle in fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=10).legend_handles:
        handle.set_sizes([125])

    # Save the plot
    if not os.path.isdir(os.path.join('..', 'figures')):
        os.makedirs(os.path.join('..', 'figures'))
    plt.savefig(os.path.join('..', 'figures', 'multinli.pdf'), bbox_inches='tight')
    print(os.path.exists(os.path.join('..', 'figures', 'multinli.pdf')))


if __name__ == '__main__':
    main()

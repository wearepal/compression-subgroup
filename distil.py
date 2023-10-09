import argparse
import os
import re

from datasets import load_dataset


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def main():

    # Define the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--attention',
        action='store_true',
        help='Whether to use attention distillation.'
    )
    parser.add_argument(
        '--hidden',
        action='store_true',
        help='Whether to use hidden state distillation.'
    )
    parser.add_argument(
        '--embedding',
        action='store_true',
        help='Whether to use embedding distillation.'
    )

    args = parser.parse_args()

    # Check the arguments
    if [args.attention, args.hidden, args.embedding] == [None, None, None]:
        raise NotImplementedError

    # Set the hyperparameters
    ATTENTION = args.attention
    HIDDEN = args.hidden
    EMBEDDING = args.embedding


    # Define the variables
    student_path = 'huawei-noah/TinyBERT_General_6L_768D'

    # Define the save path
    folder = ''
    if ATTENTION:
        folder += 'a'
    if HIDDEN:
        folder += 'h'
    if EMBEDDING:
        folder += 'e'
    
    save_path = os.path.join('models', 'tinybert_6', folder)


    # Download the models
    if not os.path.isdir(os.path.join('models', 'bert-base-uncased')):
        for file in ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']:
            os.system(f"wget -P {os.path.join('models', 'bert-base-uncased')} https://huggingface.co/bert-base-uncased/resolve/main/{file}")

    if not os.path.isdir(os.path.join('models', 'huawei-noah/TinyBERT_General_6L_768D')):
        for file in ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']:
            os.system(f"wget -P {os.path.join('models', student_path)} https://huggingface.co/{student_path}/resolve/main/{file}")        


    # Download the datasets
    if not os.path.isdir(os.path.join('data', 'wikipedia')):
        os.makedirs(os.path.join('data', 'wikipedia'))
    
    if not os.path.isfile(os.path.join('data', 'wikipedia', 'wikipedia.txt')):
        dataset = load_dataset('wikipedia', '20220301.en', split='train')
        dataset = dataset.remove_columns(['id', 'url', 'title'])
        dataset = dataset.map(lambda x: {'text': x['text'].replace('\n', ' ')})

        dataset.to_csv(os.path.join('data', 'wikipedia', 'wikipedia.txt'), sep='\n', header=False, index=False)
        with open(os.path.join('data', 'wikipedia', 'wikipedia.txt'), 'r') as file:
            filedata = file.read()

        filedata = filedata.replace('\n', '\n\n')
        with open(os.path.join('data', 'wikipedia', 'wikipedia.txt'), 'w') as file:
            file.write(filedata)
        
    if not any('.json' in file for file in os.listdir(os.path.join('data', 'wikipedia'))):
        os.system(f"""
            python {os.path.join('distillation', 'pregenerate_training_data.py')} \
                --train_corpus {os.path.join('data', 'wikipedia', 'wikipedia.txt')} \
                --bert_model {os.path.join('models', 'bert-base-uncased')} \
                --reduce_memory \
                --do_lower_case \
                --epochs_to_generate 3 \
                --output_dir {os.path.join('data', 'wikipedia')}
        """)


    # General distillation
    os.system(f"""
        python {os.path.join('distillation', 'general_distill.py')} \
            --pregenerated_data {os.path.join('data', 'wikipedia')} \
            --teacher_model {os.path.join('models', 'bert-base-uncased')} \
            --student_model {os.path.join('models', student_path)} \
            --reduce_memory \
            --do_lower_case \
            --train_batch_size 256 \
            --gradient_accumulation_steps 2 \
            --output_dir {save_path} \
            {'--attention' if ATTENTION else ''} \
            {'--hidden' if HIDDEN else ''} \
            {'--embedding' if EMBEDDING else ''}
    """)


    # Copy the final checkpoint
    files = os.listdir(save_path)
    files = [file for file in files if '.bin' in file]
    files = sorted_alphanumeric(files)
    os.system(f"cp {os.path.join(save_path, files[-1])} {os.path.join(save_path, 'pytorch_model.bin')}")
    print(f'Final checkpoint: {os.path.join(save_path, files[-1])}')

    # Remove the other checkpoints
    for file in files[:-2]:
        os.system(f'rm {os.path.join(save_path, file)}')


if __name__ == '__main__':
    main()

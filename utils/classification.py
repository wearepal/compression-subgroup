import glob
import json
import numpy as np
import os
import pandas as pd
import shutil
import torch
import torch.nn as nn

from transformers import Trainer, EvalPrediction, EarlyStoppingCallback

from loss import LossComputer


# Defined functions
def train_model(args, model, train_data, val_data, checkpoint=False):

    def compute_metrics(p: EvalPrediction):
        y_pred = torch.from_numpy(p.predictions).to('cuda')
        y_true = torch.from_numpy(p.label_ids).to('cuda')
        g = torch.from_numpy(val_data.get_group_array()).to('cuda')

        loss_computer = LossComputer(val_data, nn.CrossEntropyLoss(reduction='none'))
        _ = loss_computer.loss(y_pred, y_true, g)
        
        return {'average': loss_computer.avg_acc, 'worst': min(loss_computer.avg_group_acc)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train(resume_from_checkpoint=checkpoint)
    model.save_pretrained(args.output_dir)

    # Save the loss
    with open(os.path.join(args.output_dir, 'loss.txt'), 'w') as file:
        for obj in trainer.state.log_history:
            file.write(json.dumps(obj))
            file.write('\n\n')

    # Clean up the checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


def test_model(args, model, test_data):
    trainer = Trainer(model, args=args)

    y_pred = trainer.predict(test_data)
    y_true = test_data.get_label_array()
    g = test_data.get_group_array()

    # Save the predictions
    data = pd.DataFrame({'Prediction': np.argmax(y_pred[0], axis=1), 'Label': y_true, 'Group': g})
    data.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

    y_pred = torch.from_numpy(y_pred[0]).to('cuda')
    y_true = torch.from_numpy(y_true).to('cuda')
    g = torch.from_numpy(g).to('cuda')

    loss_computer = LossComputer(test_data, nn.CrossEntropyLoss(reduction='none'))
    _ = loss_computer.loss(y_pred, y_true, g)

    # Save the results
    loss_computer.log_stats(args.output_dir)

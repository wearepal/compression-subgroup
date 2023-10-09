import torch

from nni.algorithms.compression.pytorch.pruning import TransformerHeadPruner
from torch.nn.utils import prune

from classification import train_model


# Defined functions
def prune_model(args, model, train_data, val_data, sparsity, pruning):

    if pruning == 'unstructured':
        parameters_to_prune = [(module, 'weight') for module in filter(lambda m: type(m) == torch.nn.Linear, model.modules())]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity)

    elif pruning == 'structured':
        attention_name_groups = list(zip(
            ['encoder.layer.{}.attention.self.query'.format(i) for i in range(12)],
            ['encoder.layer.{}.attention.self.key'.format(i) for i in range(12)],
            ['encoder.layer.{}.attention.self.value'.format(i) for i in range(12)],
            ['encoder.layer.{}.attention.output.dense'.format(i) for i in range(12)]
        ))

        kwargs = {
            'ranking_criterion': 'l1_weight',
            'global_sort': True,
            'num_iterations': 1,
            'head_hidden_dim': 64,
            'attention_name_groups': attention_name_groups
        }
        
        config_list = [{
            'sparsity': sparsity,
            'op_types': ['Linear'],
            'op_names': [x for layer in attention_name_groups for x in layer]
        }]

        pruner = TransformerHeadPruner(model.base_model, config_list, **kwargs)
        pruner.compress()

        speedup_rules = {}
        for group_idx, group in enumerate(pruner.attention_name_groups):
            # get the layer index
            layer_idx = None
            for part in group[0].split('.'):
                try:
                    layer_idx = int(part)
                    break
                except:
                    continue
            if layer_idx is not None:
                speedup_rules[layer_idx] = pruner.pruned_heads[group_idx]
        
        pruner._unwrap_model()
        model.base_model._prune_heads(speedup_rules)

    train_model(args, model, train_data, val_data)

    if pruning == 'unstructured':
        for module_name, module in parameters_to_prune:
            prune.remove(module_name, module)

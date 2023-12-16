## Are Compressed Language Models Less Subgroup Robust?

Official code for the paper titled [**Are Compressed Language Models Less Subgroup Robust?**](https://aclanthology.org/2023.emnlp-main.983/) presented at **EMNLP 2023** - Main Track.

> To reduce the inference cost of large language models, model compression is increasingly used to create smaller scalable models. However, little is known about their robustness to minority subgroups defined by the labels and attributes of a dataset. In this paper, we investigate the effects of 18 different compression methods and settings on the subgroup robustness of BERT language models. We show that worst-group performance does not depend on model size alone, but also on the compression method used. Additionally, we find that model compression does not always worsen the performance on minority subgroups. Altogether, our analysis serves to further research into the subgroup robustness of model compression.

### Datasets
The MultiNLI and CivilComments datasets have to be downloaded manually and placed in the correct directory.
- **MultiNLI:** Download the dataset from [here](https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz) and place it in './data/multinli'.
- **CivilComments:** Download the dataset from [here](https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/) and place it in './data/jigsaw'.

### Training
The main.py script is used to fine-tune the compressed models on MultiNLI, CivilComments, SCOTUS, and the binary variants of MultiNLI. The script can be run in the following way:

```
CUDA_VISIBLE_DEVICES=0 python main.py --data multinli --kd bert_medium
```
Full options can be found in the argparse section of the [script](https://github.com/wearepal/compression-subgroup/blob/main/main.py).

### Distillation
The distil.py script is used to ablate the TinyBERT 6 model. The distilled models will be saved in a separate "models" folder. The script can be run in the following way:

```
CUDA_VISIBLE_DEVICES=0 python distil.py --data multinli --attention --hidden --embedding
```
Full options can be found in the argparse section of the [script](https://github.com/wearepal/compression-subgroup/blob/main/distil.py).

### Citation
```
@inproceedings{gee-etal-2023-compressed,
    title = "Are Compressed Language Models Less Subgroup Robust?",
    author = "Gee, Leonidas  and
      Zugarini, Andrea  and
      Quadrianto, Novi",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.983",
    pages = "15859--15868",
    abstract = "To reduce the inference cost of large language models, model compression is increasingly used to create smaller scalable models. However, little is known about their robustness to minority subgroups defined by the labels and attributes of a dataset. In this paper, we investigate the effects of 18 different compression methods and settings on the subgroup robustness of BERT language models. We show that worst-group performance does not depend on model size alone, but also on the compression method used. Additionally, we find that model compression does not always worsen the performance on minority subgroups. Altogether, our analysis serves to further research into the subgroup robustness of model compression.",
}
```

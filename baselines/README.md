[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Neural Baseline Models for DSTC10 Track 1

This directory contains the official baseline codes for [DSTC10 Track 1](../README.md). 


It includes an unified GPT-2 based baseline model MOD-GPT and a BERT-based model MemeBERT. 

We will provide the model structure details in Task Description Paper. If you want to publish experimental results with this dataset or use the baseline models, please cite the following article. 

**NOTE**: This paper reports the results with an earlier version of the dataset and the baseline models, which will differ from the baseline performances on the official challenge resources.



Please first put the dialogue data to the path: 
 
```
data/dialog
```
And put the extracted meme feature file to the path: 

```
data/meme
```
Download the parameters from huggingface for corresponding tasks: 

| Task                                   | Initial parameters          |
|----------------------------------------|----------------------------|
| Text Response Modeling      | [Chinese](https://huggingface.co/thu-coai/CDial-GPT2_LCCC-base), [English](https://huggingface.co/gpt2)  |
| Meme Retrieval  | [Chinese](https://huggingface.co/bert-base-chinese), [English](https://huggingface.co/bert-base-uncased) | 
| Meme Emotion Classification  | [Chinese](https://huggingface.co/thu-coai/CDial-GPT2_LCCC-base), [English](https://huggingface.co/gpt2) | 

And put the model into the path: 
```
ckpt/original_gpt
```
Finally, train the model with cmd: 
```
python train.py 
```


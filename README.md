[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# DSTC10 Track1 - MOD: Internet Meme Incorporated Open-domain Dialogue

This repository contains the data, scripts and baseline codes for [DSTC10](https://sites.google.com/dstc.community/dstc10/tracks) Track 1.

As a kind of new expression elements, Internet memes are popular and extensively used in online chatting scenarios since they manage to make dialogues vivid, moving, and interesting. Meanwhile, there are many implicit and strong emotions delivered by Internet memes. However, most current dialogue researches focus on text-only conversation tasks. This challenge track aims to support Meme incorporated Open-domain Dialogue, where the formation of utterances can be text-only, meme-only, or mixed information. Compared to previous dialogue tasks, MOD is much more challenging since it requires the model to understand the multimodal elements as well as the emotions behind them.  Correspondingly, we further split the challenge into three sub-tasks: 1) text response modeling, which evaluates the quality of text-only response, 2) meme retrieval, which discriminates the suitability of meme usage, and 3) emotion classification, which predicts the current emotion type for speakers. 

**Organizers:** Zhengcong Fei, Zekang Li, Jinchao Zhang, Yang Feng, Jie Zhou


## Important Links 

* [Track Proposal](https://drive.google.com/file/d/1uLqEuZCd3szKhXb-zSH-nlKG5Ea-nuTA/view) 
* [Data Formats](data/README.md)
* [Baseline Details](baselines/README.md) 

## Task

This challenge consists of three sub-tasks: 

| Task #1 | Text Response Modeling | 
|:---------------:|--------------------------------------------|
| Goal | To generate a coherent and natural text response given the multi-modal history context | 
| Input | Multi-modal dialogue history (u_1, u_2, ..., u_{t-1}), where u_i = (S_i, m_i), and S_i represent text-only response and m_i represents suitable meme id | 
| Output | Natural、fluent、and informative machine text response S_t in line with dialogue history |



| Task #2 | Meme Retrieval | 
|:---------------:|--------------------------------------------|
| Goal | To select a suitable internet meme from candidates given the multi-modal history context and generated text response | 
| Input | Multi-modal dialogue history (u_1, u_2, ..., u_{t-1}) and generated text response S_t, where u_i = (S_i, m_i), and S_i represent text-only response and m_i represents suitable meme id | 
| Output | Suitable and vivid internet meme m_t in line with dialogue history |



| Task #3 | Meme Emotion Classification | 
|:---------------:|--------------------------------------------|
| Goal | To predict the emotion type when respond with an internet meme | 
| Input | Multi-modal dialogue history (u_1, u_2, ..., u_t), where u_i = (S_i, m_i), and S_i represent text-only response and m_i represents suitable meme id | 
| Output | Emotion c_t for current meme usage |


Participants will develop systems to generate the outputs for each task.
They can leverage the annotations and the ground-truth responses available in the training and validation datasets.

In the test phase, participants will be given a set of unlabeled dialogue history test instances.
And they will submit **up to 5** system outputs for **all three tasks**.

**NOTE**: For someone who are interested in only one or two of the tasks, we recommend to use our baseline system for the remaining tasks to complete the system outputs.



## Evaluation 

Each submission will be evaluated in the following task-specific automated metrics first:

| Task                                   | Automated Metrics          |
|----------------------------------------|----------------------------|
| Text Response Modeling      | BLEU、Dist |
| Meme Retrieval  | Recall_n@K, MAP | 
| Meme Emotion Classification  | Accuracy@K | 


* BLEU2-4 is the English word level scoring, and dist1-2 is the automatic indicator of the diversity of dialogue content at the English word level.
* Recall_n@K measures if the positive meme is ranked in the top k positions of n candidates. Mean average precision (MAP) consider the rank order.
* The top k accuracy, referred to as Accuracy@k, indicates that if the correct emotion type in the highest k-class score emotion type, the score of metric is 1. 
* The final ranking for text response modeling will be based on **human evaluation results** only for selected systems according to automated evaluation scores. It will address the following aspects: grammatical/semantical correctness, naturalness, appropriateness, informativeness and relevance to given multimodal history. 


## Data 

In this challenge track, participants will use an augmented version of MOD dataset which includes multi-turn meme-incorporated open-domain conversions. All the ground-truth annotations for meme usage as well as the emotion stations are available to develop the components on the [training and validation sets](https://drive.google.com/drive/folders/1MCXEwNe5YcHkBtTh7S1O2g1WDznbh8z-?usp=sharing). 


In the test phase, participants will be evaluated on the results generated by their models for two testing data sets: easy version and hard version. To evaluate the generalizability of meme selection, the unseen internet memes will appear in the hard testing set.


Data and system output format details can be found from [data/README.md](data/README.md). In particular, we provide the illustration of data formation for each task in detail. 

## Timeline 

* Training data released: Jun 14, 2021 
* Test data released: Sep 15, 2021


## Rules

* Participation is welcome from any team (academic, corporate, non profit, government).
* The identity of participants will NOT be published or made public. In written results, teams will be identified as team IDs (e.g. team1, team2, etc). The organizers will verbally indicate the identities of all teams at the workshop chosen for communicating results.
* Participants may identify their own team label (e.g. team5), in publications or presentations, if they desire, but may not identify the identities of other teams.
* Participants are allowed to use any external datasets, resources or pre-trained models. 


## Contact 


### Join the DSTC mailing list to get the latest updates about DSTC10 
* To join the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/join

* To post a message: send your message to list@dstc.community

* To leave the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/unsubscribe

### For specific enquiries about DSTC10 Track1

Please feel free to contact:  feizhengcong (at) gmail (dot) com 

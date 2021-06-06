[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# DSTC10 Track1 - MOD: Internet Meme Incorporated Open-domain Dialogue

This repository contains the data, scripts and baseline codes for [DSTC10](https://sites.google.com/dstc.community/dstc10/tracks) Track 1.

As a kind of new expression elements, Internet memes are popular and extensively used in online chatting scenarios since they manage to make  dialogues vivid, moving, and interesting. This challenge track aims to support Meme incorporated Open-domain Dialogue, where the formation of utterances can be text-only, meme-only, or mixed information. Compared to previous dialogue tasks, MOD is much more challenging since it requires the model to understand the multimodal elements as well as the emotions behind them.

**Organizers:** Zhengcong Fei, Zekang Li, Jinchao Zhang, Yang Feng, Jie Zhou


## Important Links 

* [Track Proposal](https://drive.google.com/file/d/1uLqEuZCd3szKhXb-zSH-nlKG5Ea-nuTA/view) 


## Task

This challenge consists of three sub-tasks: 

| Task #1 | Text Response Modeling | 
|:---------------:|--------------------------------------------|
| Goal | To generate a coherent and natural text response given the multi-modal history context | 
| Input | multi-modal dialogue history (u_1, u_2, ..., u_{t-1}), where u_i = (S_i, m_i), and S_i represent text-only response and m_i represents suitable meme id | 
| Output | natural、fluent、and informative machine text response S_t in line with dialogue history |


## Evaluation 

Each submission will be evaluated in the following task-specific automated metrics first:

| Task                                   | Automated Metrics          |
|----------------------------------------|----------------------------|
| Text Response Modeling      | BLEU、Perplexity、Dist |


## Data 

In this challenge track, participants will use an augmented version of MOD dataset which includes multi-turn meme-incorporated open-domain conversions. All the ground-truth annotations for meme usage as well as the emotion stations are available to develop the components on the training and validation sets. 

Data and system output format details can be found from [data/README.md](data/README.md). 

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

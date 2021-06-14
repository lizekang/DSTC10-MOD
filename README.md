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




# DSTC10 赛道一 - 表情融入的开放式对话 

本项目包含了对于DSTC10 [赛道一](https://sites.google.com/dstc.community/dstc10/tracks)的数据集、脚本和基准线模型 

网络表情，作为一种新的内容表达方式，因其能使对话更加生动、有趣，在网络聊天场景中得到了广泛的使用。同时，表情可以传递更多隐含信息和情感。然而，当前大多数对话研究专注于文本形式。本次评测专注于结合网络表情的开放领域对话任务。其中，对话回复的形式可以是文本，表情，或两者形式的结合。与以前的文本对话任务相比，融入表情的对话建模更具有挑战性，它要求模型理解图片文本等多模态因素及它们背后的情感。 对此，我们进一步划分为三个子任务：根据给定的多模态对话历史， 1） 文本回复建模，评测文本回复生成的质量 2）表情检索，确定网络表情使用的合理性，和 3）情感分类，预测当前对话者的情感状态。

**组织者:** 费政聪, 李泽康, 张金超, 冯洋, 周杰 


## 重要链接 

* [赛道提出](https://drive.google.com/file/d/1uLqEuZCd3szKhXb-zSH-nlKG5Ea-nuTA/view) 
* [数据格式](data/README.md)
* [基准线](baselines/README.md) 

## Task

本次挑战由三个子任务组成：

| 任务 #1 | 文本回复建模 | 
|:---------------:|--------------------------------------------|
| 目标 | 生成一致和自然的文本回复，给定多模态历史内容 | 
| 输入 | 多模态对话历史 (u_1, u_2, ..., u_{t-1}), 其中 u_i = (S_i, m_i), S_i 表示文本回复， m_i 表示合适的表情序号 | 
| 输出 | 符合对话历史的自然、流畅和丰富的文本回复 S_t |



| 任务 #2 | 表情检索 | 
|:---------------:|--------------------------------------------|
| 目标 | 从表情候选集中选择合适的网络表情，根据给定的多模态历史和生成的文本回复 | 
| 输入 | 多模态对话历史(u_1, u_2, ..., u_{t-1}) 和生成的文本回复 S_t, 其中 u_i = (S_i, m_i), S_i 表示文本回复, m_i 表示合适的表情序号 | 
| 输出 | 符合对话历史的、合适的表情序号 m_t |



| 任务 #3 | 情感分类 | 
|:---------------:|--------------------------------------------|
| 目标 | 预测情感类别，当使用网络表情的时候 | 
| 输入 | 多模态对话历史 (u_1, u_2, ..., u_t), 其中 u_i = (S_i, m_i), S_i 表示文本回复, m_i 表示合适的表情序号 | 
| 输出 | 当前对话者的情感 c_t |


参与者被邀请搭建对话系统来对每个子任务产生输出。
在训练集和验证集中给出了情感标记和正确的多模态回复。

在测试阶段，参与者会得到没有标注的对话历史测试样例。
参与者被允许提交最多5词输出，对于每个子任务。


**注意**: 对于那些只对个别任务感兴趣的参与者, 建议使用我们提供的基准系统来完成剩下的任务。



## 评测

对于每个提交,首先都会被以下自动指标进行评测: 


| 任务                                   |自动指标          |
|----------------------------------------|----------------------------|
| 文本回复建模      | BLEU、Dist |
| 表情检索  | Recall_n@K, MAP | 
| 情感分类  | Accuracy@K | 


* BLEU2-4是中文字级别打分，Dist1-2是中文字级别的对话内容多样性的自动指标.
* Recall_n@K测量是否正确的表情出现在前k个位置，给定n个候选。平均精度均值(Mean average precision)则考虑了结果中的排序顺序.
* top k 准确率表示在多分类情况下最高的k类得分表情标签，与真实值相同，则认为得分为1. 
* 对于文本回复建模任务最终排名基于人工评估。它会评测以下方面：语法/语义正确性，对话自然性，对话合适性和与多模态历史的相关性。



## 数据 

在本次挑战中，参与者将使用扩增版的MOD数据集合，它包括了多轮表情融入的开放式对话。所有表情使用和情感状态标注是提供的，在[训练集和验证集](https://drive.google.com/drive/folders/1MCXEwNe5YcHkBtTh7S1O2g1WDznbh8z-?usp=sharing)中. 


在测试阶段，参与者将被要求提供两个测试集合的结果用于表现评测：容易版本和困难版本。其中，困难版本测试集出现了没有在训练集出现的网络表情，用于评测模型的泛化能力。

数据和模型输出格式可以参考 [data/README.md](data/README.md). 特别地, 我们提供了每个子任务格式的详细说明. 

## 时间线 

* 训练集和验证集释放: 2021年6月14日 
* 测试集释放: 2021年9月15日


## 联系 

### 询问具体细节关于 DSTC10 赛道1

欢迎联系:  feizhengcong (at) gmail (dot) com  


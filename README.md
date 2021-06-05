[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# DSTC10 Track1 - MOD: Internet Meme Incorporated Open-domain Dialogue

This repository contains the data, scripts and baseline codes for [DSTC10](https://sites.google.com/dstc.community/dstc10/tracks) Track 1.

As a kind of new expression elements, Internet memes are popular and extensively used in online chatting scenarios since they manage to make  dialogues vivid, moving, and interesting. This challenge track aims to support Meme incorporated Open-domain Dialogue, where the formation of utterances can be text-only, meme-only, or mixed information. Compared to previous dialogue tasks, MOD is much more challenging since it requires the model to understand the multimodal elements as well as the emotions behind them.

**Organizers:** Zhengcong Fei, Zekang Li, Jinchao Zhang, Yang Feng, Jie Zhou


## Task

## Evaluation 


## Data
### How to get the dataset 

We release the train/valid data set on google drive and two test version sets will be used online challenging leaderboard.  

### Copyright 

The original copyright of all the conversations belongs to the source owner.
The copyright of annotation belongs to our group, and they are free to the public.
The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes and distributed to others.

 
### Data Sample 


|  Json Key Name  | Description                                |
|:---------------:|--------------------------------------------|
| dialogue xxxxx  | current dialogue id                        |
| speaker_id      | speaker id                                 |
| emotion_id      | emotion type                               |
| image_id        | id of internet meme set                    |
| txt             | text-only response                         |



```json
{
    "dialogue 43992": [
        {
            "speaker_id": "[speaker1]",
            "emotion_id": 0,
            "img_id": "195",
            "txt": "\u53ef\u4ee5\u7684\u5ba2\u5b98\u62cd\u4e0b\u8bf4\u4e00\u58f0\u8981\u624b\u52a8\u6539\u4ef7"
        },
        {
            "speaker_id": "[speaker2]",
            "txt": "\u90a3\u6211\u4e70\u4e24\u4efd\u4e24\u4e2a\u5730\u5740"
        },
        {
            "speaker_id": "[speaker1]",
            "emotion_id": 1,
            "img_id": "272",
            "txt": "\u5ba2\u5b98\u8fd9\u4e2a\u662f\u5728\u540c\u4e00\u5730\u5740\u4e24\u4e2a\u5730\u5740\u4e0d\u884c\u54e6"
        },
        {
            "speaker_id": "[speaker2]",
            "txt": "\u6211\u7684\u610f\u601d\u662f\u6211\u4e70\u516d\u888b"
        } 
    ]
}
```

## Timeline 
* Training data released: Jun 14, 2021 
* Test data released: Sep 15, 2021


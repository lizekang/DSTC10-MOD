# DSTC10 Track1 Dataset 

This directory contains the official training/validation datasets for [DSTC10 Track 1](../README.md).  


### How to get the dataset 

We release the train/valid data set on google drive and two test version sets will be used online challenging leaderboard.  


### Training Data Sample 

We provide the data formation illustration for the training set. Each dict in the JSON file represents one dialogue sample, which contains multi-turn meme incorportated utterences.   


|  Json Key Name  | Description                                |
|:---------------:|--------------------------------------------|
| dialogue xxxxx  | current dialogue id                        |
| speaker_id      | speaker id for current utterence           |
| emotion_id      | emotion type for current meme usage        |
| image_id        | id of internet meme in the meme set        |
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


## Data sample for Text Response Modeling 



```json
[
    {
        "history": [
            {
                "speaker_id": "[speaker1]",
                "emotion_id": 5,
                "img_id": "030",
                "txt": "\u4eca\u5929\u5192\u96e8\u4e0a\u73ed\uff01\uff01\uff01\uff01"
            },
            {
                "speaker_id": "[speaker2]",
                "img_id": "107",
                "txt": "\u54c8\u54c8\uff5e\u6211\u4eca\u5929\u4f11\uff5e\uff5e"
            }
        ],
        "answer": {
            "speaker_id": "[speaker1]",
            "txt": "\u660e\u5929\u4ec0\u4e48\u73ed\uff1f"
        }
    },
]
```



## Data sample for Meme Retrieval 


```json
[
    {
        "history": [
            {
                "speaker_id": "[speaker1]",
                "emotion_id": 5,
                "img_id": "030",
                "txt": "\u4eca\u5929\u5192\u96e8\u4e0a\u73ed\uff01\uff01\uff01\uff01"
            },
            {
                "speaker_id": "[speaker2]",
                "img_id": "107",
                "txt": "\u54c8\u54c8\uff5e\u6211\u4eca\u5929\u4f11\uff5e\uff5e"
            },
            {
                "speaker_id": "[speaker1]",
                "txt": "\u660e\u5929\u4ec0\u4e48\u73ed\uff1f"
            }
        ],
        "candidate": {
            "speaker_id": "[speaker1]",
            "set": [
                "013",
                "023",
                "040",
                "044",
                "097",
                "110",
                "136",
                "179",
                "196",
                "202",
                "265"
            ]
        },
        "answer": {
            "speaker_id": "[speaker1]",
            "img_id": "196"
        }
    },
]
```


## Data sample for Meme Emotion Classification


```json
[
    {
        "history": [
            {
                "speaker_id": "[speaker1]",
                "emotion_id": 5,
                "img_id": "030",
                "txt": "\u4eca\u5929\u5192\u96e8\u4e0a\u73ed\uff01\uff01\uff01\uff01"
            },
            {
                "speaker_id": "[speaker2]",
                "img_id": "107",
                "txt": "\u54c8\u54c8\uff5e\u6211\u4eca\u5929\u4f11\uff5e\uff5e"
            },
            {
                "speaker_id": "[speaker1]",
                "txt": "\u660e\u5929\u4ec0\u4e48\u73ed\uff1f",
                "img_id": "196"
            }
        ],
        "answer": {
            "speaker_id": "[speaker1]",
            "emotion_id": 29
        }
    },
]
```



### Copyright 

The original copyright of all the conversations belongs to the source owner.
The copyright of annotation belongs to our group, and they are free to the public.
The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes and distributed to others.

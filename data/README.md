# DSTC10 Track1 Dataset 

This directory contains the official training/validation datasets for [DSTC10 Track 1](../README.md).  


## How to get the dataset 

We release the train/valid data set on [google drive](https://drive.google.com/drive/folders/1MCXEwNe5YcHkBtTh7S1O2g1WDznbh8z-?usp=sharing) and two test version sets will be used online challenging leaderboard.  

The data divided into the following three subsets: 

* Training set 
  * c_train.json: Chinese training instances 
  * e_train.json: English training instances 
* Validation set 
  * c_val.json: Chinese validation instances 
  * c_val_task1.json: Chinese validation instances for task 1: Text Response Modeling 
  * c_val_task2.json: Chinese validation instances for task 2: Meme Retrieval 
  * c_val_task3.json: Chinese validation instances for task 2: Meme Emotion Classification 
  * e_val.json: English validation instances 
  * e_val_task1.json: English validation instances for task 1: Text Response Modeling 
  * e_val_task2.json: English validation instances for task 2: Meme Retrieval 
  * e_val_task3.json: English validation instances for task 2: Meme Emotion Classification 
* Supplementary file: 
  * img2id.json: project the name of meme set into img id 
  * emotion_dict: project the emotion type into emotion id 


Participants will develop systems to take the key "history" of *c/e_val_task.json* as an input and generates the outputs following the **same format** as the key "answer" to estimate the performance.


## Training Data Sample 

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

In this task, the conversation system is supposed to generate a coherent and natural text-only "answer" given the multi-modal "history". 


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

In this task, the system is supposed to select a suitable "internet meme id" from "candidates" given the multi-modal "history" context and generated text-only "answer".

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

In this task, the conversation system is supposed to predict the "emotion id" when respond with an "internet meme id". 


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

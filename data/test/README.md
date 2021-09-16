# DSTC10 Track1 Test Dataset 

This directory contains the official testing datasets for [DSTC10 Track 1](../README.md).  

We release the testing data set on [google drive](https://drive.google.com/drive/folders/1HXAQBJeaMnxb7oj-sfyqtkbX_NEjY5BZ?usp=sharing). 

The data divided into the following four subsets: 

* c_test_easy: Chinese testing set for easy verison 
* c_test_hard: Chinese testing set for hard verison 
* e_test_easy: English testing set for easy verison 
* e_test_hard: English testing set for hard verison 

The data formats for each task can be [seen](../README.md). 

For Text Response Modeling, please replace the "txt" value in answer with your answers. 

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
            "txt": ""
        }
    },
]
```

For Meme Retrieval, please replace the "img_id" value in answer with you results, i.e., a sorted list for candidate set. For example, \["013", "040", "023", "044", "097", "110", "136", "179", "202", "196", "265"\].  

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
            "img_id": ""
        }
    },
]
```

For Meme Emotion Classification, please replease "emotion_id" value with your answer, i.e., top-5 candidate emtion ids. 

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
            "emotion_id": ""
        }
    },
]
```

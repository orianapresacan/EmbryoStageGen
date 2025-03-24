# Classification

## Data

Add the data in the `data` directory following the structure below. 1,000 real images per class were used for training, 100 real images for testing, and 100 synthetic images for validation. In the case of synthetic images, 5,000 images per class were used to augment the real dataset.
```  
data/
├── real/
│   ├── train/
│   │   ├── 2cell/
│   │   ├── 4cell/
│   │   ├── 8cell/
│   │   ├── Blastocyst/
│   │   └── Morula/
│   ├── validation/
│   │   ├── 2cell/
│   │   ├── 4cell/
│   │   ├── 8cell/
│   │   ├── Blastocyst/
│   │   └── Morula/
│   └── test/
│       ├── 2cell/
│       ├── 4cell/
│       ├── 8cell/
│       ├── Blastocyst/
│       └── Morula/
└── synthetic/
    ├── GAN/
    │   ├── 2cell/
    │   ├── 4cell/
    │   ├── 8cell/
    │   ├── Blastocyst/
    │   └── Morula/
    └── LDM/
        ├── 2cell/
        ├── 4cell/
        ├── 8cell/
        ├── Blastocyst/
        └── Morula/
```
The real dataset can be found [here](https://zenodo.org/records/14253170) and the synthetic dataset [here](https://drive.google.com/file/d/1egpag71fUtZTcB04Bn4mLeVo5s2jh9-W/view?usp=drive_link).

The checkpoints for our trained classification models can be downloaded from [this link](https://drive.google.com/drive/folders/1UkpWeBqZlxUJ08KJxnIi-LNMhgfmWOSh?usp=drive_link).

Use the _train()_ function in the `train.py` file to train the models, and the _test()_ function to evaluate them and obtain the metrics.

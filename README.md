```
.
├── data
├── generator
│   └── det
│       ├── defects.py
│       ├── edge_cases.py
│       ├── generator.py
│       ├── geometry.py
│       ├── layouts             # 12 layouts + 1 base layout files
│       └── run.py
├── model                       # Model Architecture
│   ├── det                     # DBNet++ (Text Detection)
│   │   ├── backbone.py
│   │   ├── dbnet.py
│   │   ├── dcn.py
│   │   ├── head.py
│   │   ├── layers.py
│   │   ├── loss.py
│   │   ├── neck.py
│   └── rec                     # SVTRv2 (Text Recognition)
│       ├── loss.py
│       ├── svtr_ctc.py
│       ├── svtr_encoder.py
│       ├── tokenizer.py
│       └── vocab.py
├── READ.md
├── src                         # Path for training/validating/testing
│   ├── det
│   │   ├── dataloader.py
│   │   ├── DBNetPP.ipynb
│   │   ├── test.py
│   │   ├── train.py
│   │   └── val.py
│   ├── preprocess              # Using U-2-Net (pre-trained) to remove background
│   │   ├── scanner.ipynb
│   │   └── scanner.py
│   └── rec2
│       ├── dataloader.py
│       ├── train.py
│       └── val.py
├── tests                       # Folder to test some function, classes
└── weights
    ├── det
    │   ├── best_model.pth
    │   ├── checkpoint_epoch_*.pth
    │   └── training_log.csv
    └── rec2_aug

```
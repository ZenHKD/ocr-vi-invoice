```
.
├── data
│   ├── test
│   ├── train_det
│   ├── train_rec
│   ├── val_det
│   └── val_rec
├── generator
│   ├── det
│   │   ├── defects.py
│   │   ├── edge_cases.py
│   │   ├── generator.py
│   │   ├── geometry.py
│   │   ├── layouts             # 12 layouts + 1 base layout files
│   │   └── run.py
│   └── rec
│       ├── fonts
│       ├── run.py
│       └── text_generator.py
├── model                       # Model Architecture
│   ├── det                     # DBNet++ (Text Detection)
│   │   ├── backbone.py
│   │   ├── dbnet.py
│   │   ├── dcn.py
│   │   ├── head.py
│   │   ├── layers.py
│   │   ├── loss.py
│   │   ├── neck.py
│   └── rec                     # SVTR-CTC (Text Recognition)
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
│   └── rec
│       ├── dataloader.py
│       ├── train.py
│       └── val.py
├── tests                       # Folder to test some function, classes
└── weights
    ├── det
    │   ├── best_model.pth
    │   ├── checkpoint_epoch_10.pth
    │   └── training_log.csv
    └── rec
```
```
.
├── data
│   ├── test/                   # Test images copy from `data/archive/val_images/val_images`
|   ├── archive/                # MC_OCR 2021 data
|   ├── SROIE2019/              # SROIE data
|   ├── test/
|   ├── test_det_sroie/
|   ├── train_det/
|   ├── train_rec/
|   ├── val_det_sroie/
|   ├── val_rec/
|   ├── vietocr                 # VietOCR data
|   |   ├── en_00
|   |   ├── en_01
|   |   ├── InkData_line_processed
|   |   ├── meta
|   |   ├── random
|   |   ├── vi_00
|   |   └── vi_01
|   └── vinh/                   # Folder to save some images take from real life 
├── generator
│   └── det
│       ├── defects.py
│       ├── edge_cases.py
│       ├── generator.py
│       ├── geometry.py
│       ├── layouts/            # 12 layouts + 1 base layout files
│       └── run.py
├── model                       # Model Architecture
│   ├── det                     # DBNet++ (Text Detection) (pre-trained ResNet-50-dcn in backbone)
│   │   ├── backbone.py
│   │   ├── dbnet.py
│   │   ├── dcn.py
│   │   ├── head.py
│   │   ├── layers.py
│   │   ├── loss.py
│   │   └── neck.py
│   └── rec2                    # SVTRv2 (Text Recognition) (simplified version, trained from scratch)
│       ├── loss.py
│       ├── svtrv2.py
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
│   ├── pipeline
│   │   ├── pipeline.ipynb      # Script to pass and check inference of all models
│   │   └── pipeline2.py
│   ├── preprocess              # Using U-2-Net (pre-trained) to remove background (use directly from `rembg`)
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
    ├── rec2                    # Folder save the trained SVTRv2 model without augmentation (function `get_train_augmentation` in src/rec2/dataloader.py)
    │   ├── best_model.pth
    │   ├── checkpoint_epoch_*.pth
    │   └── training_log.csv
    └── rec2_aug                # Folder save the trained SVTRv2 model with augmentation
        ├── best_model.pth
        ├── checkpoint_epoch_*.pth
        └── training_log.csv
```

# Data preparation

## Text Detection 

We synthetic training set with 12 layouts (12 files in generator/det/layouts) which contains 20,000 images, then use the [MC_OCR 2021](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) as the validation set and test set (their training set -> our validation set (626 samples), their test set -> our test set (347 samples)).

We use this command to synthetic:
```bash
python -m generator.det.run --num 20000 --scenario training_hard --output data/train_det
```

The result is:
```
Sample Types:
    blank: 1630 (8.2%)
    edge_case: 7041 (35.2%)
    realistic: 9909 (49.5%)
    unreadable: 1420 (7.1%)

Layout Types:
    cafe_minimal: 1496 (7.5%)
    delivery_receipt: 1563 (7.8%)
    ecommerce_receipt: 1268 (6.3%)
    formal_vat: 1214 (6.1%)
    handwritten: 1296 (6.5%)
    hotel_bill: 1287 (6.4%)
    modern_pos: 1437 (7.2%)
    restaurant_bill: 1588 (7.9%)
    supermarket_thermal: 2309 (11.5%)
    taxi_receipt: 941 (4.7%)
    traditional_market: 1299 (6.5%)
    utility_bill: 1252 (6.3%)
```

## Text Recognition

We use the training data from [VietOcr](https://github.com/pbcquoc/vietocr), the data is given by author [see it here](https://drive.google.com/file/d/1T0cmkhTgu3ahyMIwGZeby612RpVdDxOR/view) which contains  601,282 samples of 7 categories: en_00, en_01, InkData_line_processed, meta, random, vi_00, vi_01. The validation set and test set are taken from [MC_OCR 2021](https://www.kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021). We use their text_recognition_train_data.txt as our validation set (5,285 samples) and text_recognition_val_data.txt as test set (1,300 samples).


# Model & Trainning Strategy

## Preprocess (U-2 Net)

The purpose of this model is removing background for cleaner input for text detection model.

## Text Detection (DBNet++)

Model was trained with 30 epochs .The model was freezed in backbone (ResNet50-DCN) for first 5 epochs, then be full fine-tuned for the rest of epochs. 

The best epoch is Epoch: 16 (see in `weights/det/training_log.csv`)
Test result for the test set:

```
==================================================
Test Results (SROIE):
==================================================
Loss:      1.5046
Precision: 0.7282
Recall:    0.8123
F1 Score:  0.7659
IoU:       0.6230
Dice:      0.7659
==================================================
```

## Text Recognition (SVTRv2-base) (self-implement, simplified version)

At the begining, the model was trained with 30 epochs (see in `weights/rec2/training_log.csv`) with no augmentation (no function `def get_train_augmentation()` in `src/rec2/dataloader`). The best result was at epoch 25.

Then, load the best_model.pth from `weghts/rec2` to continue fine-tuning with augmentatiom for 20 epochs. The best result now is at epoch 12 (see in `weights/rec2_aug/training_log.csv`).

Test result for the test set:

```
Test Loss: 1.0536 | Test CER: 0.1626 | Test Accuracy: 0.2431
```
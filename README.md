# Swin-Unet Based Nuclei Instance Segmentation and Classification

Deep learning pipeline for detecting, separating, and classifying nuclei in histopathology images using a Swin-Unet based segmentation model and center-voting based instance separation.


## Dataset
The PanNuke dataset is a large-scale pathology dataset designed for nuclear instance segmentation and classification, containing annotated nuclei from 19 different tissue types. It consists of 7,904 histopathology image patches and 205,343 annotated nuclei, with each nucleus having an instance segmentation mask and a corresponding cell-type label.

The dataset is divided into three folds, with each fold containing approximately 2,600 images and around 70,000 annotated nuclei. The nuclei are categorized into five classes:
```
Neoplastic
Non-neoplastic epithelial
Inflammatory
Connective
Dead cells
```
Images are processed at:

```
256 × 256 × 3
```




## Challenge

Nuclei in histopathology images frequently touch or overlap.

A standard binary segmentation model can identify the nuclear region but may merge multiple touching nuclei into a single connected object.


```text
Two touching nuclei as separate instances

   11111112222222
 11111111112222222
 11111111112222222
   11111112222222


Where:

1 = Nucleus 1
2 = Nucleus 2
```
Generated Mask from Segmentation Model
```text
Two touching nuclei

   █████████████
 █████████████████
 █████████████████
   █████████████

```




## Solution

The model uses a Swin-Unet backbone with three output heads:

Mask — predicts nuclear probability for every pixel
H — predicts horizontal displacement toward the nucleus center
V — predicts vertical displacement toward the nucleus center

The H/V predictions are converted into center votes during inference.

### Segmentation model training


```text
                         Input Image
                              |
                              v
                 +-----------------------+
                 |   Swin-Tiny Encoder   |
                 +-----------------------+
                              |
                              v
                 +-----------------------+
                 |    U-Net Decoder      |
                 +-----------------------+
                              |
                              v
                 +-----------------------+
                 |   Shared Features     |
                 +-----------------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
        +-----------+   +-----------+   +-----------+
        | Mask Head |   |  H Head   |   |  V Head   |
        +-----------+   +-----------+   +-----------+
              |               |               |
              v               v               v
        Mask Prediction     H_pred          V_pred
              |               |               |
              |               +-------+-------+
              |                       |
              v                       v
        BCE + Dice Loss       Center Consistency
                                     Loss
                                      |
                                      v
                               Backpropagation


```

### Center Consistency Loss
```
                 H_pred                    V_pred
                    |                        |
                    +-----------+------------+
                                |
                                v
                    +-----------------------+
                    |  Predicted Center     |
                    |                       |
                    | x_pred = x + H_pred   |
                    | y_pred = y + V_pred   |
                    +-----------+-----------+
                                |
                                |
              Ground Truth     |     Predicted
              Instance Map     |     Center
                    |           |
                    v           v
          +----------------------------------+
          | Group pixels belonging to        |
          | the same nucleus instance        |
          +----------------+-----------------+
                           |
                           v
          +------------------------------------------+ |
          | Centroid of  predicted centroid for each   |
          | nucleus                                    |
          +----------------+-------------------------+ |
                           |
                           |
                           v
                 +---------------------+  |
                 | Mean Squared Error     |
                 |       (MSE)            |
                 |   ((X-X_pred)/**2) /2  |
                 +---------+-------------+
                           |
                           v
                 Center Consistency
                       Loss
                           |
                           v
                    Backpropagation

```



## Model

### Swin-Unet

The segmentation model uses a Swin Transformer encoder with a U-Net style decoder.

The decoder produces shared spatial features that are passed to three prediction heads:
``` text

                 Shared Features
                       |
          +------------+------------+
          |            |            |
       Mask Head     H Head       V Head
          |            |            |
       Mask P(x,y)    H(x,y)       V(x,y)
```

### Mask Head

Predicts whether each pixel belongs to a nucleus.

### H/V Heads

Predict the displacement from each nucleus pixel toward its corresponding nucleus center.

These fields provide the additional geometric information required to separate touching nuclei.

### Center Voting

For every predicted nuclear pixel (x, y), the model predicts:
``` text
H → horizontal displacement
V → vertical displacement
```
The displacement is converted into a predicted center location:
``` text
center_x = x + 256 × H
center_y = y + 256 × V
```
Every foreground pixel therefore produces a vote for where it believes the nucleus center is.

For pixels belonging to the same nucleus:

``` text
pixel 1 ──┐
pixel 2 ──┤
pixel 3 ──┼──> same center
pixel 4 ──┤
pixel 5 ──┘
```

For two touching nuclei, the desired behavior is to produce two different vote clusters:

```
             Center 1          Center 2

                ●                 ●
             ● ● ●             ● ● ●
           ● ● ● ● ●         ● ● ● ● ●
             ● ● ●             ● ● ●
                ●                 ●
```
This allows a connected nuclear mask to be converted into separate instances.

## Inference Pipeline

After training, the model produces:
```
Mask + H + V
```
### 1. Generate Nuclear Mask

The predicted mask probability is thresholded to obtain foreground nuclear pixels.

### 2. Generate Center Votes

Each predicted foreground pixel is converted into a predicted center using its H/V values.

### 3. Build Vote Map

All center votes are accumulated into a 2D vote map.

Dense regions in the vote map represent likely nucleus centers.

### 4. Detect Centers

The vote map is smoothed and local maxima are detected.

Each strong peak represents a predicted nucleus center.

### 5. Build Instance Map

Each predicted nuclear pixel is assigned to the closest detected center based on its predicted center vote.

The resulting instance map contains:
```
0 → background
1 → nucleus 1
2 → nucleus 2
3 → nucleus 3
```

This allows touching nuclei to be separated without relying only on connected components.

## Classification

Once individual nucleus instances are available, the learned Swin features are used for nucleus classification.

Instead of using a single pixel or simple RGB statistics, features from the entire nucleus region are aggregated.

### Classification model training
```



             TRAINING IMAGE
                    |
                    v
          +-------------------+
          | Trained Swin      |
          | Encoder           |
          +-------------------+
                    |
                    v
             Swin Feature Maps
                    |
                    + <--------- Ground-truth
                    |             Instance Mask
                    v
          +-------------------+
          | Masked Feature    |
          | Pooling            |
          +-------------------+
                    |
                    v
          Nucleus Embedding
              (1344-D)
                    |
                    + <--------- Ground-truth
                    |             Nucleus Class
                    v
          +-------------------+
          | Classification    |
          | FC Neural Network |
          +-------------------+
                    |
                    v
              Predicted Class
                    |
                    v
            Cross-Entropy Loss
                    |
                    v
              Backpropagation

```

For each nucleus instance, the corresponding instance mask is projected onto the Swin feature map and used to pool the features inside that nucleus.

This produces one feature vector representing the complete nucleus.

The representation captures information such as:

```
Nucleus shape
Texture
Appearance
Local spatial context
Learned semantic features
```

The resulting feature vector is passed to the classification head to predict the nucleus type.

## Loss

The segmentation model uses a multi-task loss:
```
Total Loss =
    Mask BCE
  + Mask Dice
  + 2 × H Loss
  + 2 × V Loss
  + Center Consistency Loss
```
### Mask BCE

Binary cross-entropy trains the mask head to distinguish nuclear pixels from background.

### Dice Loss

Dice loss improves overlap between predicted and ground-truth nuclear regions.

### H/V Loss

Regression losses train the horizontal and vertical displacement fields.

```
H prediction → horizontal center direction
V prediction → vertical center direction
```

### Center Consistency Loss

For each nucleus instance, center votes are calculated for its pixels.

Votes belonging to the same nucleus are encouraged to form a compact cluster.

```
Before:

    •              •
          •
  •                    •
             •

After training:

        • • •
      •   C   •
        • • •

```

The instance_map ensures that votes from different touching nuclei are handled as separate groups during training.

## Results
```
Current segmentation results:
| Metric              |      Result |
| ------------------- | ----------: |
| Dice                |   0.801 |
| H Correlation       |   0.838 |
| V Correlation       |   0.862 |
| Mean Center Error   | 2.41 px |
| Median Center Error | 2.14 px |
| Mean Vote Spread    | 6.36 px |
| Median Vote Spread  | 5.03 px |
```


## Pixel-level Results
```
GT nucleus pixels        : 170,560
Predicted nucleus pixels : 214,318
Correct nucleus pixels   : 156,806
```

The model learns meaningful center-directed displacement fields rather than simply predicting values close to zero.

## Center Vote Analysis

For the evaluated nuclei:
```
Mean center error   : 2.41 px
Median center error : 2.14 px

Mean vote spread    : 6.36 px
Median vote spread  : 5.03 px
```

Example:

Nucleus 4
```


GT center        : (192.27, 112.38)
Predicted center : (188.89, 110.95)

Center error     : 3.68 px
Vote spread      : 6.44 px
```

Nucleus 5
```
GT center        : (247.09, 233.77)
Predicted center : (246.06, 234.27)

Center error     : 1.15 px
Vote spread      : 3.28 px
```
The center-vote representation is designed to produce compact, instance-specific vote clusters that can be used to separate touching nuclei.

## Project Structure
```
.
├── models/
│   └── swin_unet.py
│
├── dataset/
│   └── dataset.py
│
├── losses/
│   └── losses.py
│
├── training/
│   └── train.py
│
├── inference/
│   ├── center_voting.py
│   ├── instance_segmentation.py
│   └── classification.py
│
├── evaluation/
│   └── metrics.py
│
├── visualization/
│   └── visualize.py
│
├── checkpoints/
│
├── requirements.txt
└── README.md
```
## Tech Stack
```
Python
PyTorch
Swin Transformer
U-Net
NumPy
SciPy
scikit-image
Matplotlib
```

## Key Features

```
Swin Transformer based Encoder for feature extraction
U-Net style decoder
Three-head nuclear segmentation [Mask, H ,V]
H/V center displacement prediction
Center-voting based instance separation
Designed to handle touching nuclei
Instance-level feature extraction
Swin feature based nucleus classification
Center-vote visualization and analysis
Quantitative segmentation and center metrics
```

## Pipeline Summary

```
Histopathology Image
        ↓
Swin-Unet
        ↓
Mask + H + V
        ↓
Nuclear Mask + Center Votes
        ↓
Center Detection
        ↓
Instance Map
        ↓
Individual Nuclei
        ↓
Swin Feature Pooling
        ↓
Nucleus Classification
```


The core approach combines Swin-based visual features, nuclear segmentation, H/V center prediction, and center-vote based instance separation to produce individual nucleus instances that can subsequently be classified.

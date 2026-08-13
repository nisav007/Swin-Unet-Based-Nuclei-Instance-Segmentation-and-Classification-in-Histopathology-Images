# Swin-Unet Based Nuclei Instance Segmentation and Classification

Deep learning pipeline for detecting, separating, and classifying nuclei in histopathology images using a Swin-Unet based segmentation model and center-voting based instance separation.

## Problem

Nuclei in histopathology images frequently touch or overlap.

A standard binary segmentation model can identify the nuclear region but may merge multiple touching nuclei into a single connected object.

```text
Two touching nuclei

   █████████████
 █████████████████
 █████████████████
   █████████████
```

For downstream analysis, we need:

```text
   AAAAAAABBBBBBB
 AAAAAAAAAABBBBBBB
 AAAAAAAAAABBBBBBB
   AAAAAAABBBBBBB
```

where each nucleus is represented as a separate instance and can then be classified.

## Solution

The model uses a Swin-Unet backbone with three output heads:

Mask — predicts nuclear probability for every pixel
H — predicts horizontal displacement toward the nucleus center
V — predicts vertical displacement toward the nucleus center

The H/V predictions are converted into center votes during inference.
```text
Image
  |
  v
Swin Transformer + U-Net Decoder
  |
  +------ Mask
  |
  +------ H
  |
  +------ V
           |
           v
      Center Votes
           |
           v
      Center Detection
           |
           v
      Instance Map
           |
           v
   Individual Nuclei
           |
           v
   Swin Feature Pooling
           |
           v
     Classification

```

The pipeline has two main stages:

Instance segmentation — detect and separate individual nuclei.
Classification — use Swin features from each detected nucleus to predict its class.

## Dataset
The PanNuke dataset is a semi-automatically generated pathology dataset for nuclear instance segmentation, comprehensively covering nuclear labels of 19 different tissue types. The dataset contains a total of 7,904 images and 205,343 annotated nuclei, each with an instance segmentation mask and corresponding cell type labels (tumor epithelial cells, inflammatory cells, connective tissue cells) 
```
256 × 256
```
The training pipeline uses:

Histopathology image
Binary nuclear mask
Instance-level nucleus map
Horizontal displacement map
Vertical displacement map
Nucleus class labels for classification

Images are processed at:


## Model

Swin-Unet

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

Mask Head

Predicts whether each pixel belongs to a nucleus.

H/V Heads

Predict the displacement from each nucleus pixel toward its corresponding nucleus center.

These fields provide the additional geometric information required to separate touching nuclei.

Center Voting

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

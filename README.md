# Swin-Unet Based Nuclei Instance Segmentation and Classification

Deep learning pipeline for detecting, separating, and classifying nuclei in histopathology images using a Swin-Unet based segmentation model and center-voting based instance separation.

## Problem

Nuclei in histopathology images often touch or overlap.

A standard binary segmentation model can correctly identify the nuclear region but may produce:

```text
Two touching nuclei

   █████████████
 █████████████████
 █████████████████
   █████████████

as a single connected object.

For downstream analysis, we need:

   AAAAAAABBBBBBB
 AAAAAAAAAABBBBBBB
 AAAAAAAAAABBBBBBB
   AAAAAAABBBBBBB

where each nucleus is represented as a separate instance and can then be classified.
''' text

## Solution

The model predicts three outputs from the same Swin-Unet backbone:

Mask — nuclear probability for every pixel
H — horizontal displacement toward the nucleus center
V — vertical displacement toward the nucleus center

The H/V fields are converted into center votes during inference.

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

This separates the problem into two practical stages:

Instance segmentation: find and separate individual nuclei.
Classification: use the learned Swin features from each nucleus instance to predict its class.
Dataset

The dataset contains histopathology images with nucleus annotations.

The training pipeline uses:

Histopathology image
Binary nuclear mask
Instance-level nucleus map
Horizontal displacement map
Vertical displacement map
Nucleus class labels for classification

Images are processed at:

256 × 256

The instance map assigns a unique ID to each nucleus:

0 → background
1 → nucleus 1
2 → nucleus 2
3 → nucleus 3
...

The instance IDs are used during training to calculate center consistency separately for each nucleus.

Model
Swin-Unet

The segmentation model uses a Swin Transformer encoder with a U-Net style decoder.

The decoder produces a shared feature representation which is passed to three prediction heads:

                 Shared Features
                       |
          +------------+------------+
          |            |            |
       Mask Head     H Head       V Head
          |            |            |
       Mask P(x,y)    H(x,y)       V(x,y)
Mask Head

Predicts whether each pixel belongs to a nucleus.

H/V Heads

Predict the displacement from each nucleus pixel toward its corresponding nucleus center.

These fields provide the information needed to separate touching nuclei.

Center Voting

For every predicted nuclear pixel (x, y), the model predicts a displacement:

H → horizontal displacement
V → vertical displacement

The displacement is converted into a predicted center location:

center_x = x + 256 × H
center_y = y + 256 × V

Therefore, every nucleus pixel produces a vote for where it believes the nucleus center is.

For pixels belonging to the same nucleus:

pixel 1 ──┐
pixel 2 ──┤
pixel 3 ──┼──> same center
pixel 4 ──┤
pixel 5 ──┘

For two touching nuclei, the pixels should generate two different vote clusters:

             Center 1          Center 2

                ●                 ●
             ● ● ●             ● ● ●
           ● ● ● ● ●         ● ● ● ● ●
             ● ● ●             ● ● ●
                ●                 ●

This allows a connected nuclear mask to be converted into separate instances.

Inference Pipeline

The trained model produces:

Mask + H + V
1. Generate nuclear mask

The mask probability is thresholded to obtain predicted nuclear pixels.

2. Generate center votes

Each foreground pixel is converted into a predicted center using its H/V values.

3. Build vote map

All center votes are accumulated into a 2D vote map.

4. Detect centers

The vote map is smoothed and local maxima are detected.

Each strong peak represents a predicted nucleus center.

5. Build instance map

Each predicted nuclear pixel is assigned to the closest detected center based on its predicted center vote.

The result is:

0 → background
1 → nucleus 1
2 → nucleus 2
3 → nucleus 3
...

This step allows touching nuclei to be separated without relying only on connected components.

Classification

Once individual nucleus instances are available, the model's Swin features are used for nucleus classification.

Instead of classifying a single pixel, features from the entire nucleus region are aggregated.

Swin Feature Map
       |
       +---- Predicted Nucleus Mask
                    |
                    v
           Masked Feature Pooling
                    |
                    v
             Nucleus Feature
                    |
                    v
            Classification Head
                    |
                    v
              Nucleus Class

For an instance mask M
k
	​

, the features inside that nucleus are pooled to produce a single feature vector representing the complete nucleus.

This preserves information about:

Nucleus shape
Texture
Appearance
Local spatial context
Learned semantic features from the Swin encoder

The classification stage can then predict the nucleus type for every detected instance.

Loss

The segmentation model uses a multi-task loss:

Total Loss =
    Mask BCE
  + Mask Dice
  + 2 × H Loss
  + 2 × V Loss
  + Center Consistency Loss
Mask BCE

Binary cross-entropy trains the model to distinguish nuclear pixels from background.

Dice Loss

Dice loss improves overlap between the predicted and ground-truth nuclear masks, particularly under foreground/background imbalance.

H/V Loss

Regression losses train the horizontal and vertical displacement fields.

H prediction → horizontal center direction
V prediction → vertical center direction
Center Consistency Loss

For each nucleus instance, the center vote from every pixel is calculated.

The votes belonging to the same instance are encouraged to agree with each other.

Conceptually:

Before:

    •              •
          •
  •                    •
             •

After training:

        • • •
      •   C   •
        • • •

The instance_map is used during training so that votes from two different touching nuclei are treated as separate groups.

Results

Current segmentation results:

Metric	Result
Dice	0.801
H MAE	0.0200
V MAE	0.0192
H Correlation	0.838
V Correlation	0.862
Mean Center Error	2.41 px
Median Center Error	2.14 px
Mean Vote Spread	6.36 px
Median Vote Spread	5.03 px
Pixel-level results
GT nucleus pixels       : 170,560
Predicted nucleus pixels: 214,318
Correct nucleus pixels  : 156,806

The H/V predictions perform substantially better than predicting zero displacement:

                 H MAE       V MAE

Zero baseline    0.0385      0.0369
Model            0.0200      0.0192

This indicates that the network is learning meaningful center-directed displacement fields rather than simply predicting values close to zero.

Center Vote Analysis

For the evaluated nuclei:

Mean center error   : 2.41 px
Median center error : 2.14 px

Mean vote spread    : 6.36 px
Median vote spread  : 5.03 px

Example:

Nucleus 4

GT center       : (192.27, 112.38)
Predicted center: (188.89, 110.95)

Center error    : 3.68 px
Vote spread     : 6.44 px

Another example:

Nucleus 5

GT center       : (247.09, 233.77)
Predicted center: (246.06, 234.27)

Center error    : 1.15 px
Vote spread     : 3.28 px

The goal of the center-vote representation is not just accurate centroid prediction, but producing compact, instance-specific vote clusters that can be used to separate touching nuclei.

Project Structure
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
Tech Stack
Python
PyTorch
Swin Transformer
U-Net
NumPy
SciPy
scikit-image
Matplotlib
Key Features
Swin Transformer based feature extraction
U-Net style decoder
Multi-head nuclear segmentation
H/V center displacement prediction
Center-voting based instance separation
Designed for touching nuclei
Instance-level feature extraction
Swin feature based nucleus classification
Quantitative center-vote analysis
Visualization of predicted centers and vote distributions
Pipeline Summary
Histopathology Image
        ↓
Swin-Unet
        ↓
Mask + H + V
        ↓
Nuclear Mask
        +
Center Votes
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

The main focus of the project is using center-directed H/V predictions to turn a semantic nuclear mask into an instance-level segmentation, followed by nucleus-level classification using the learned Swin representations.

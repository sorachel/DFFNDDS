# DFFNDDS

DFFNDDS: PREDICTION OF SYNERGISTIC DRUG COMBINATIONS WITH DUAL FEATURE FUSION NETWORKS

The official PyTorch implementation of DFFNDDS: Prediction of Synergistic Drug Combinations with Dual Feature Fusion Networks. DFFNDDS utilizes a fine-tuned pretrained language model and dual feature fusion mechanisms to predict synergistic drug combinations.

![Image](https://user-images.githubusercontent.com/92193559/192967888-d3e85614-f441-4465-a1ad-e949f570bbf5.png)

## requirements:

DFFNDDS requires:
* torch : 1.10.2
* cuda :  11.3
* python : 3.8.12

## Steps:
1. The input SMILES string is encoded by a fine-tuned BERT model, converting the features into vectors.
2. The output of the SMILES encoder, hashed atom pair fingerpints and gene expressions of cancer cell lines are input into the projector module, mapping the inputs to the same dimension.
3. To fuse the features, we utilize two networks (multi-head attention mechanism and highway network) to extract and combine the input features in the dual fusion block.
4. The outputs of the two networks are concatenated to obtain the final feature representation, which is propagated through the linear layer.

# Projet détection arrhythmie par TCN et SSL

Ce projet traîte des données ecg patients via un modèle Temporal Convolutional Network en Self-Supervised Learning.

Lien dataset : https://physionet.org/content/mitdb/1.0.0/#files-panel

### Pipeline : 

- Données brutes MIT-BIH
- Augmentation : bruit gaussien + scaling
- SSL - SimCLRModel : NT-Xent loss sur paires (z1, z2)
- Encodeur pré-entraîné - représentations h        
- LinearProbe
- FineTunedModel

Donc pipeline en 2 phases : 

**Phase 1 — Self-Supervised Learning (SSL)**
L'encodeur TCN est pré-entraîné sans labels via SimCLR :

On prend un signal ECG, on crée 2 vues augmentées (x1, x2)
L'objectif est que les représentations z1 et z2 du même signal soient proches, et éloignées des autres signaux du batch
La NT-Xent loss guide ça — aucun label de classe n'est utilisé

**Phase 2 — Supervised Learning**
On prend l'encodeur pré-entraîné et on l'évalue/affine avec les labels :

Linear Probing : on gèle l'encodeur, on entraîne juste une couche linéaire → supervisé minimal
Fine-tuning : on dégeèle tout et on entraîne avec les 5 classes → supervisé complet
Baseline : un TCN entraîné from scratch, 100% supervisé, sans pré-entraînement SSL

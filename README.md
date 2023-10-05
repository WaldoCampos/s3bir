# S3BIR - Self-Supervised Sketch-Based Image Retrieval

This repository contains different ways of creating Sketch-Based Image Retrieval models trained using the Self-Supervised paradigm.

**WARNING:** Every path in the repository is hard-coded and has to be replaced accordingly.

To start training a model, create a configuration file and use it with train.py

``python train.py --config your_config.ini``

To evaluate a model, use eval.py with the same configuration file

``python eval.py --config your_config.ini``

The repository can be tested with Quick, Draw! data in the following colab notebook: https://colab.research.google.com/drive/1fdCY5X4969sJW_uoE2H9vQCdjJqW9ula?usp=sharing

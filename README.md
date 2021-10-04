# Lung Histopathological Image Classifier

The primary goal of this project was to produce a deep learning solution that classifies lung histopathological images into three classes (benign, adenocarcinomas, and squamous cell carcinomas). The model is trained with transfer learning, using MobileNetV2. The new model has been retrained on a dataset containing 15,000 RGB lung histopathological images, evenly split into three classes.

Interpretability using saliency mapping was planned to be used in order to visually explain which features the model was looking at in determining which of the three classes it belongs to. The goal was to show a heatmap of an image with the brightest part being the feature that best explains the classification of an image by the model. This part of the project was not able to be completed during the project lifecycle.

The project can be deployed as an IPython Notebook to be run locally in a Jupyter Notebook or as a Google Colab instance for accessibility to users who may not have the necessary hardware to run the model locally.

This project was run on:

• Python 3.6
• Tensorflow 2.6

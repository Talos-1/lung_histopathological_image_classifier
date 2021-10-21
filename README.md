# Lung Histopathological Image Classifier

Run the project online with [Binder](https://mybinder.org/v2/gh/Talos-1/lung_histopathological_image_classifier/HEAD)

The primary goal of this project was to produce a deep learning solution that classifies lung histopathological images into three classes (benign, adenocarcinomas, and squamous cell carcinomas). The model is trained with transfer learning, using [MobileNetV2](https://github.com/Talos-1/lung_histopathological_image_classifier/blob/main/Sandler%20M%20et%20al.%2C%20MobileNetV2%20-%20Inverted%20Residuals%20and%20Linear%20Bottlenecks.pdf). The new model has been retrained on a dataset containing 15,000 RGB lung histopathological images, evenly split into three classes.

Interpretability with saliency mapping was planned to be used in order to visually explain which features the model was looking at in determining which of the three classes it belongs to. The goal was to show a heatmap of an image with the brightest part being the feature that best explains the classification of an image by the model. This part of the project was not able to be completed during the project lifecycle.

The project can be deployed as an IPython Notebook (`.ipynb`) to be run locally in a Jupyter Notebook or as a Google Colab instance for accessibility to users who may not have the necessary hardware to run the model locally. A `.py` file has also been added if a IPython environment is not available.

This project was run on:

• Python 3.6

• Tensorflow 2.6


## References

[1] A. Koul et al., Practical Deep Learning for Cloud, Mobile and Edge: Real-World AI and Computer Vision Projects Using Python, Keras and TensorFlow, 1st ed. O'Reilly Media, 2019, Chapter 3: Cats versus Dogs: Transfer Learning in 30 Lines with Keras.

[2] Borkowski A et al., Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019

[3] DeepLearning.AI Deep Learning Specialisation, Coursera.

[4] Deep Lizard, Keras - Python Deep Learning Neural Network API. 2021.

[5] Sandler M et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks. arXiv:1801.04381v4 [eess.IV], 2019

[6] Stanford Machine Learning Course, Coursera.

[7] Stanford University, Lecture 5 | Convolutional Neural Networks. 2017.

[8] Stanford University, Lecture 9 | CNN Architectures. 2017.

[9] T. Wood, "Convolutional Neural Network", DeepAI, 2021. [Online]. Available: https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network.

[10] "TensorFlow Core v2.6.0", TensorFlow, 2021. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf.

[11] Zhang A et al., Dive into Deep Learning, Release 0.17.0. O'Reilly Media, 2021

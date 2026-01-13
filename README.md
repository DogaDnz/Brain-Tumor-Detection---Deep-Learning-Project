# Brain-Tumor-Detection---Deep-Learning-Project
A Deep Learning project using CNN to classify Brain MRI scans into 4 categories (Glioma, Meningioma, Pituitary, and No Tumor) with 87% accuracy using TensorFlow/Keras.

## Brain Tumor MRI Classification using CNN
This project implements a Deep Learning model to classify Brain MRI scans into four distinct categories. By leveraging a custom Convolutional Neural Network (CNN) architecture built with TensorFlow and Keras, the model achieves a high diagnostic accuracy to assist in medical imaging analysis.

## Performance Summary
The model was evaluated on a comprehensive test set of 1,311 images, achieving an overall accuracy of 87%.

Final Classification Report:

<img width="690" height="325" alt="image" src="https://github.com/user-attachments/assets/c04fe3e6-fd23-49dd-8cf0-da3d7588f559" />


Key Insights:
* High Reliability in Healthy Scans: A recall of 0.99 for "No Tumor" ensures that almost all healthy patients are correctly identified.

* Excellent Pituitary Detection: The model achieves a 0.96 F1-score for Pituitary tumors, showing strong feature extraction for this class.

* Challenge Area: Class 1 (Meningioma) presents the lowest recall (0.61), indicating visual similarities with other tumor types that require further model tuning or more data.

### Tech Stack & Tools
* Language: Python

* Deep Learning: TensorFlow 2.x, Keras

* Data Handling: NumPy, Pandas, ImageDataGenerator

* Visualization: Matplotlib, Seaborn

* Evaluation: Scikit-learn (Classification Report, Confusion Matrix)

  ### Model Architecture
  The network follows a sequential design focused on hierarchical feature extraction:
  1) Input Layer: $224 \times 224 \times 3$ RGB images.
  Convolutional Blocks: Multiple Conv2D layers with increasing filters (32, 64, 128) and ReLU activation.
  2) Downsampling: MaxPooling2D layers to reduce spatial dimensions and prevent overfitting.
  3) Regularization: A Dropout (0.5) layer is implemented after the Flatten layer to minimize co-dependency between neurons.
  4) Output Layer: A Dense layer with 4 neurons and Softmax activation for multi-class probability distribution.
 
      ### Training Process
     To improve the model's generalization, the following techniques were applied:
     * Data Augmentation: Random rotations 20 and horizontal flips were used to simulate diverse clinical scenarios.
     * Normalization: All pixel values were rescaled to the $[0, 1]$ range.Validation Strategy: 20% of the training data was used as a validation set to monitor Validation Accuracy and Validation Loss during training.


<img width="1628" height="608" alt="image" src="https://github.com/user-attachments/assets/5dc69823-6833-4b34-953f-fde806d2f3ca" />


### Dataset
The model was trained on the Brain Tumor MRI Dataset available on Kaggle. It includes:

* Glioma

* Meningioma

* No Tumor

* Pituitary

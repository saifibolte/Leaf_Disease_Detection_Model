# Leaf Disease Detection Model through Transfer Learning
Plant leaf diseases are a major threat to global food security and the environment, causing significant losses in crop yields and biodiversity. Advanced agricultural and modern equipment technology has made humans capable of producing sufficient food as per the all species requirement. However, there are several threats to food security including climate change, pollinator‘s declination and plant disease etc. Unhealthy Plants due to any disease is a major problem for food security and it leads to an unfavourable outcome to farmers whose livelihoods depend on a healthy crop only. Accurate detection of these diseases is therefore important for a variety of applications, including crop protection, pest control, and environmental monitoring. In recent years, convolutional neural networks (CNNs) have emerged as a powerful tool for solving this problem, achieving state-of-the-art results on a variety of datasets. However, there are still limitations to these models, including the need for large amounts of labelled data and the risk of overfitting.

The whole project is based on the training and testing of the taken data sets using the two main approaches i.e. training from scratch and transfer learning approach. For training and transfer learning, VGG19 is used. The best accuracy result is shown as the output

## Dataset Used: 
This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

### Link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/

## Building CNN:
The approach involves training a CNN on a dataset of plant leaf images and corresponding disease labels. It addresses the issue of class imbalance in the dataset by using a combination of oversampling and weighting techniques, which allows the model to better recognize the minority class. It uses a combination of stochastic gradient descent and Adam optimization algorithms to train the model, which allows for faster convergence and better performance. We have trained our model on the Plant Village dataset using Google Colab In-built GPU. Also Transfer learning significantly reduces training time and gives much better performance for a relatively small dataset.

## Training the model: 
Training the model had occurred using the transfer learning approach and early stopping and model checkpoints had been set. The steps per epoch had been set to 64, number of epochs set to 50 and validation steps also set on 64.

Model loading takes place after that as the name “best_model.h5” and after that evaluating the model had occurred with the accuracy of 87.09%.

## Testing the model: 
The trained model is tested on a set of images. Random images are introduced to the network and the output label is compared to the original known label of the image.

## Definition of generalization?

In machine learning, generalization is a definition to demonstrate how well is a trained model to classify or forecast unseen data. Training a generalized machine learning model means, in general, it works for all subset of unseen data. An example is when we train a model to classify between dogs and cats. If the model is provided with dogs images dataset with only two breeds, it may obtain a good performance. But, it possibly gets a low classification score when it is tested by other breeds of dogs as well. This issue can result to classify an actual dog image as a cat from the unseen dataset. Therefore, data diversity is very important factor in order to make a good prediction. In the sample above, the model may obtain 85% performance score when it is tested by only two dog breeds and gains 70% if trained by all breeds. However, the first possibly gets a very low score (e.g. 45%) if it is evaluated by an unseen dataset with all breed dogs. This for the latter can be unchanged given than it has been trained by high data diversity including all possible breeds.

It should be taken into account that data diversity is not the only point to care in order to have a generalized model. It can be resulted by nature of a machine learning algorithm, or by poor hyper-parameter configuration. In this post we explain all determinant factors. There are some methods (regularization) to apply during model training to ensure about generalization. But before, we explain bias and variance as well as underfitting and overfitting.

## Variance and bias (overfitting and underfitting)

Variance and bias are two important terms in machine learning. Variance means the variety of predictions values made by a machine learning model (target function). Bias means the distance of the predictions from the actual (true) target values. A high-biased model means its prediction values (average) are far from the actual values. Also, high-variance prediction means the prediction values are highly varied.

### Variance-bias trade-off

The prediction results of a machine learning model stand somewhere between a) low-bias, low-variance, b) low-bias, high-variance c) high-bias, low-variance, and d) high-bias, high-variance. A low-biased, high-variance model is called overfit and a high-biased, low-variance model is called underfit. By generalization, we find the best trade-off between underfitting and overfitting so that a trained model obtains the best performance. An overfit model obtains a high prediction score on seen data and low one from unseen datsets. An underfit model has low performance in both seen and unseen datasets.

![img](https://deepai.space/wp-content/uploads/2020/07/sphx_glr_plot_underfitting_overfitting_001-1024x366.png)Three models with underfitting (left), goodfit (middle), and overfitting (center). Credit: https://scikit-learn.org/

![img](https://deepai.space/wp-content/uploads/2020/07/800px-Overfitting_svg.svg_.png)Overfitting/overtraining in supervised learning (e.g., neural network). Training error is shown in blue, validation error in red, both as a function of the number of training cycles. If the validation error increases(positive slope) while the training error steadily decreases(negative slope) then a situation of overfitting may have occurred. The best predictive and fitted model would be where the validation error has its global minimum. Credit: Wikipedia user: Gringer. Source: https://en.wikipedia.org/wiki/Overfitting

## Determinant factors to train generalized models

There are different ways to secure that a machine learning model is generalized. Below we explain them.

### Dataset

In order to train a classifier and generate a generalized machine learning model, a used dataset should contain diversity. It should be noted that it doesn’t mean a huge dataset but a dataset containing all different samples. This helps classifier to be trained not only from a specific subset of data and therefore, the generalization is better fulfilled. In addition, during training, it is recommended to use cross validation techniques such as K-fold or Monte-Carlo cross validations. These techniques better secure to exploit all possible portions of data and to avoid generating an overfit model.

### Machine Learning algorithm

Machine learning algorithms differently act against overfitting, underfitting. Overfitting is more likely with nonlinear, non-parametric machine learning algorithms. For instance, Decision Tree is a non-parametric machine learning algorithms, meaning its model is more likely with overfitting. On the other hand, some machine learning models are too simple to capture complex underlying patterns in data. This cause to build an underfit model. Examples are linear and logistic regression.

### Model complexity

When a machine learning models becomes too complex, it is usually prone to overfitting. There are methods that help to make the model simpler. They are called Regularization methods. Following we explain it.

#### Regularization

Regularization is collection of methods to make a machine learning model simpler. To this end, certain approaches are applied to different machine learning algorithms, for instance, pruning for decision trees, dropout techniques for neural networks, and adding a penalty parameters to the cost function in Regression.
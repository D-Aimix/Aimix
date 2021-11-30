## What is Machine Learning

Machine learning is a field of artificial intelligence and is the ability of machines to automatically learn from experience without being explicitly programmed in the same way the human do. 

Types of Traditional ML Paradigms are listed below. For now, is sufficient to know the types, but if you feel like it go ahead and read the description of each heading!



## Types of Traditional Machine Learning Algorithms

### Supervised Learning

Supervised learning is a main type of machine learning algorithms. In such algorithms, a machine learning model learns from a set of input-output pairs (known as training data) and a model is built. The model is then used to map inputs to correct outputs based on the learnt relations. Supervised learning is divided into two main types, classification and Regression. Classification is a supervised learning task focusing on the discrete data, e.g. detecting class of cars (sedan, coupe, etc.) based on the its body type, engine type, passenger capacity, etc. A model built from this type is called classifier. Regression is a supervised learning task focusing on the continuous data e.g. time-series stock forecasting. A model built from this type is called regressor.

*Warning: images might cause confusion*

### Unsupervised Learning

**Unsupervised learning** (UL) is a type of algorithm that learns patterns from untagged data. The hope is that through mimicry, the machine is forced to build a compact internal representation of its world. In contrast to [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning) (SL) where data is tagged by a human, eg. as “car” or “fish” etc, UL exhibits self-organization that captures patterns as neuronal predelections or probability densities. The other levels in the supervision spectrum are [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_Learning) where the machine is given only a numerical performance score as its guidance, and [Semi-supervised learning](https://en.wikipedia.org/wiki/Semi-supervised_learning) where a smaller portion of the data is tagged. Two broad methods in UL are Neural Networks and Probabilistic Methods.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Cluster-2.svg/601px-Cluster-2.svg.png "The result of a cluster analysis shown as the coloring of the squares into three clusters.")

### Semi-supervised Learning

**Semi-supervised learning** is an approach to [machine learning](https://en.wikipedia.org/wiki/Machine_learning) that combines a small amount of [labeled data](https://en.wikipedia.org/wiki/Labeled_data) with a large amount of unlabeled data during training. Semi-supervised learning falls between [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) (with no labeled training data) and [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) (with only labeled training data).

Unlabeled data, when used in conjunction with a small amount of labeled data, can produce considerable improvement in learning accuracy. The acquisition of labeled data for a learning problem often requires a skilled human agent (e.g. to transcribe an audio segment) or a physical experiment (e.g. determining the 3D structure of a protein or determining whether there is oil at a particular location). The cost associated with the labeling process thus may render large, fully labeled training sets infeasible, whereas acquisition of unlabeled data is relatively inexpensive. In such situations, semi-supervised learning can be of great practical value. Semi-supervised learning is also of theoretical interest in machine learning and as a model for human learning.



### Reinforcement Learning

**Reinforcement learning** (**RL**) is an area of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) concerned with how [intelligent agents](https://en.wikipedia.org/wiki/Intelligent_agent) ought to take [actions](https://en.wikipedia.org/wiki/Action_selection) in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) and [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning).

Reinforcement learning differs from supervised learning in not needing labelled input/output pairs be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).

The environment is typically stated in the form of a [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP), because many reinforcement learning algorithms for this context use [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) techniques. The main difference between the classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the MDP and they target large MDPs where exact methods become infeasible.



Interested in more jargon? Check out the detailed version, [here](https://deepai.space/machine-learning-deep-learning-algorithms/#Reinforcement_Learning)


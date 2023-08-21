# Readme--DA-DAN

DA-DAN: A Dual Adversarial Domain Adaption Network for Unsupervised Non-overlapping Cross-domain Recommendation.

The datasets used in this paper are Movie, Book, Kitchen, Food domains from [Amazon data](http://jmcauley.ucsd.edu/data/amazon/).

1. `DA_DAN.py` is the Python code that stores the model.
2. `train.py` A is the code for training the model, saving the model weight file and obtaining the result; 
3. `DADAN_Datasets.py`  is the code for loading training data. This code separates the mixed interaction sequence of users in the two domains, and scrambles the order to realize the non overlapping scenario as far as possible. The data type of the test set is the sequence of users in the target domain, and the label is the label of the source domain..
4. `NN.py`  is a comparison method. This method uses the source domain user interaction sequence on the source domain to directly train a model to act on the target domain without any domain migration.
5. `utils.py` is the test function code for testing recall and MRR indicators
6. `POP.py` is a comparison method, which gives a recommendation list according to the popularity of all items in the a domain, and the recommendation list of all users is the same.
## Requirements  
python 3.5+  
pytorch 1.11.0  
## Usage
`python train.py --gamma --Lambda --eta` for train and test DA-DAN.  
`python  train.py test` for is used to load the model weight file and test.  
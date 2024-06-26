# BillPredictor

This project builds upon previous research in predicting bill passage using machine learning, seen in Sunchoon Park's paper [Predicting the Passage of Bill Using Machine
Learning: Big Data Analysis of Factors Influencing
the Probablility of Passage of Bill](https://romanpub.com/resources/13.%20Predicting%20the%20Passge%20of%20Bill%20Using%20Machine%20Learning%20Big%20Data.pdf). We extend Park's analysis in two primary ways:

1. We add a deep learning component, deploying a neural network with two hidden layers to predict final bill passage
2. Using the direct text of previous bills put on the floor in the House of Representatives in our analysis to predict partisan lean of a bill

The intended workflow of our project is briefly explained here:
1. Pull the data and use the gradient boosting model to generate partisan lean predictions using `gradientmodel.py`
2. Tune the hyperparameters of the neural network with the csvs given from step 1 with `tune_nn.py`
3. Train and evaluate the neural network on the test set with the hyperparameters given in step 2 with `train_eval.py`, saving the model to a pth file
4. Load the trained model and predict the status of bills using both `gradientmode.py` and `eval.py`

## Data

The training and testing data is pulled using the `GovText.py` module, and has the following columns (in order):

* Number of Committees
* Number of Cosponsors
* Number of Sponsors
* Full Text
* Passed
* Partisan Lean
* *Prediction* (column added AFTER the random forest model)

We chose each column for the following reasons:

**Number of Committees**: If a bill is referred to multiple committees, attempts to reconcile the bill between the numerous committees and various hearings will take longer, leading to the bill having decreasing chances of passing.

**Number of Cosponsors**: The number of cosponsors of a bill can indicate its support in the broader House of Representatives for the bill's **general idea**, but not necessarily the exact specifics of the bill.

**Number of Sponsors**: The number of sponsors of a bill can indicate how many members have contributed substantively to the bill, which can give a sense of how the information surrounding a bill has been spread. For example, if a bill is shrouded in secrecy and only worked on by party leadership, we would expect it not just to have less sponsors but also for members to know less about the bill overall, decreasing its chance of passage.

**Full Text**: Analyzing the full text of the bill can give us an idea of the partisan lean - beyond just what party is proposing the bill - because right-wing and left-wing causes use different terminology to address the same ideas.

**Passed**: This is the target - did the bill pass or not?

## Code Explanation and Use

In this repository, there are several python scripts that correspond to each of the parts of the project. Below, they are listed and their intended use is explained.

### GovText.py

This module is a module for pure utility - it is not designed to be run from the command line. Rather, it is meant to be imported in other python scripts to be utilized. This module generates the data necessary for inputs to both the random forest model for partisan lean prediction and the neural network for bill passage prediction. This module calls the congress.gov API to get information about the bill including the bill's text, cosponsors, committees referred to, sponsors, partisan lean (for training), and the status of the House majority during that congress.

### gradientmodel.py

This script is meant to train the random forest model and generate partisan lean predictions on a test set. There are several **ordered** arguments that should be employed as follows:

* num_bills: The number of bills you would like to download for the training set. For our purposes, we used 500, which took about an hour to scrape and download.

Example use of this script is as follows:

`python3 gradientmodel.py 500`

This command trains the random forest model with a train size of 0.6 * 500 bills, and a test size of 0.4 * 500 bills. It creates two csv files - 'Training Data.csv' and 'Testing Data.csv', both of which have the partisan lean true values and the partisan lean prediction values.

### tune_nn.py

This script is meant to perform hyperparameter tuning on the neural network model using Optuna, a package that employs bayesian hyperparameter tuning. The three hyperparameters tuned are Learning Rate, Hidden Layer Size 1, Hidden Layer Size 2. There are several **ordered** arguments that should be employed as follows:

* train_path: The path to the training csv generated by `gradientmodel.py`
* val_path: The path to the testing csv generated by `gradientmodel.py`
* num_trials: The number of hyperparameter trials to generate. For the purposes of our testing, we used 50 trials.

Example use of this script is as follows:

`python3 tune_nn.py 'data/train.csv' 'data/val.csv' 50`

This command performs hyperparameter tuning with the train.csv file and the val.csv file, printing the best hyperparameter configuration.

### train_eval.py

This script is meant to train the neural network model to predict whether a bill will pass or not. It uses the Adam Stochastic Gradient Descent optimizer. It also saves the model's weights, intended to be used in `eval.py`.  There are several **ordered** arguments that should be employed as follows:

* train_path: The path to the training csv generated by `gradientmodel.py`.
* test_path: The path to the testing csv generated by `gradientmodel.py`.
* num_epochs: The desired number of epochs for testing
* lr: The desired learning rate, ideally provided by `tune_nn.py`
* hidden_size_1: The desired size of hidden layer 1, ideally provided by `tune_nn.py`
* hidden_size_2: The desired size of hidden layer 2, ideally provided by `tune_nn.py`
* save_path: The path to save the trained model's weights - note that this file **must end with .pth**

Example use of this script is as follows:

`python3 train_eval.py 'data/train.csv' 'data/test.csv' 100 0.005 14 16 'model_trained.pth'

This command trains a model and tests it using the 'train.csv' and 'test.csv' files, respectively. It trains the model for 100 epochs with a learning rate of 0.005, a hidden layer 1 size of 14, and a hidden layer size of 16. It then takes the model weights and saves it to 'model_trained.pth'

### eval.py

This script is meant to take a trained neural network, generated by `train_eval.py`, and perform evaluation using it. There are several **ordered** arguments that should be employed as follows:

* model_path: The path to the .pth file of the saved model
* hidden_size_1: The size of hidden layer 1
* hidden_size_2: The size of hidden layer 2
* test_path: The path to the file you would like to predict on

Example use of this script is as follows:

`python3 eval.py 'model_saved.pth' 14 16 'data/test.csv'`

This command evaluates a trained model using the 'test.csv' file and downloads the predictions in a file called 'Test with Predictions.csv'

## Results 
* Partisan Lean prediction — Test set Accuracy: ~82%, indicating strong performance. Given that this was a supervised learning task, the accuracy was evaluated by comparing the predicted labels to the actual partisan lean labels. This high level of accuracy suggests that the model effectively generalized from the training set without overfitting.
* Hyperparameter optimisation — the optuna module used a bayesian sweep and model plot indicated that the actual hyperparameters did not significantly impact the model results. However, the the configuration suggested by Optuna was as follows: Hidden Layer 1 Size = 13; Hidden Layer Size 2 = 16; Learning Rate = 0.00181 (note that with the Adam optimizer, the learning rate is modified for each parameter, taking into account first and second order moments.
* Model Predictor — Test set Accuracy: ~99.79%, indicating strong performance. This implies that a bill’s passage can be predicted with relative accuracy. We operationalized several non-quantitative ideas such as partisanship, member leadership, and other factors
through metrics associated with previous bills, and, using deep learning and other probabalistic approximation methods, managed to capture non-linear relationships to predict whether a bill in the US congress can pass.









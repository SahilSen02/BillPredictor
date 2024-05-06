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

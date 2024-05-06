# BillPredictor

This project builds upon previous research in predicting bill passage using machine learning, seen in Sunchoon Park's paper [Predicting the Passage of Bill Using Machine
Learning: Big Data Analysis of Factors Influencing
the Probablility of Passage of Bill](https://romanpub.com/resources/13.%20Predicting%20the%20Passge%20of%20Bill%20Using%20Machine%20Learning%20Big%20Data.pdf). We extend Park's analysis in two primary ways:

1. We add a deep learning component, deploying a neural network with two hidden layers to predict final bill passage
2. Using the direct text of previous bills put on the floor in the House of Representatives in our analysis to predict partisan lean of a bill

In this repository, there are several python scripts that correspond to each of the parts of the project. Below, they are listed and their intended use is explained.

## GovText.py

This module is a module for pure utility - it is not designed to be run from the command line. Rather, it is meant to be imported in other python scripts to be utilized. 

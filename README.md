# Diabetes Prediction

> Onset of diabetes binary classification.

Partial implementation of MLP as described in this [paper](https://www.researchgate.net/profile/Tuelay_Yildirim2/publication/228615564_Medical_diagnosis_on_Pima_Indian_diabetes_using_general_regression_neural_networks/links/541926760cf2218008bf5181/Medical-diagnosis-on-Pima-Indian-diabetes-using-general-regression-neural-networks.pdf).

Dataset - [pima-indians-diabetes](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)

Multilayer perceptron neural network with 8 inputs in the visible layer, 32 and 16 neurons in the 2 hidden layer with ReLu activation function and 1 neuron in the output layer with sigmoid activation function.

Network trained for 700 epochs with batch size of 10 using ADAM optimizer and binary_crossentropy loss function.


**Attribute Information:**

1. Number of times pregnant
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)
9. Class variable (0 or 1)

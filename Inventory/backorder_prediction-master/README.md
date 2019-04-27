# Backorder Prediction
Predicting Backorders in Inventory Mangement Context. Original dataset from Kaggle's "Can You Predict Product Backorders?", available on https://www.kaggle.com/tiredgeek/predict-bo-trial.

## Requirements
The following softwares/packages are required for running the scripts:
- Python 3.6.1
- Scikit-learn 0.19.0
- Imbalanced-learn 0.2.1

## Scripts
Before running the scripts, unzip the file data/kaggle/kaggle.rar. Order and description of scripts execution:
1. preprocessing.py - processes the original dataset, transforming the original attributes
into numeric attributes and saving into a smaller and consolidated file.
2. model_selection.py - evaluates clasification models through exhaustive grid search, using stratified 
5-fold cross-validation, and Area Under Precision-Recall Curve (AUPRC) scorer.
3. results.py - compute metrics and generate graphs for model evaluation and importance analysis.

## Reference
R. B. Santis, E. P. Aguiar and L. Goliatt, "Predicting Material Backorders in Inventory Management using Machine Learning," 4th IEEE Latin American Conference on Computational Intelligence, Arequipa, Peru, 2017.

Available from: https://www.researchgate.net/publication/319553365_Predicting_Material_Backorders_in_Inventory_Management_using_Machine_Learning

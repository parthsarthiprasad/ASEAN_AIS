# -*- coding: utf-8 -*-

# Importing files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import pickle


col_names = [	'Length',	'Beam', 'DWT',	'BHP']


populationds = pd.read_csv("./shipdata.csv", 
                            header=None, names=col_names)
# populationds.drop(['ID',	'Image',	'Vessel Type',		'Draft',
# 	'Year',	'Refit Year',	'BP',	'Class',	'Flag',
#     	'Location',	'Price US$'], axis=1, inplace=True)
# populationds['Length'] = populationds['Length'].str.replace('m', '').astype(float)
# populationds['Beam'] = populationds['Beam'].str.replace('m', '').astype(float)
# populationds['Length'] = populationds['Length'].str.replace('m', '').astype(float)



populationds = populationds.iloc[1:]
populationds.head()

#split dataset in features and target variable

feature_cols = [ 'Beam']
X = populationds[feature_cols] # Features
y = populationds.DWT # Target variable

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.50, random_state=72)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(
    solver='liblinear', random_state=72, multi_class='auto', verbose=1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy Calculations
wrong = 0
total = 0
for i in range(0, len(cm)):
    for j in range(0, len(cm)):
        total += cm[i][j]
        if i == j:
            wrong += cm[i][j]
accuracy = 100*((total-wrong)/total)
accuracy = round(accuracy, 2)

# Printing Results Optional
print('Confusion Matrix')
# print(pd.DataFrame(cm))
# print()
print(f'Out of {total} Rows, {total-wrong} were predicticted correctly')
print(f'Which Gives a Prediction Accuracy of {accuracy}%')

# iter  1 act 3.806e+01 pre 3.326e+01 delta 1.403e+00 f 5.199e+01 |g| 5.892e+01 CG   2
# iter  2 act 6.223e+00 pre 5.080e+00 delta 1.403e+00 f 1.393e+01 |g| 1.479e+01 CG   2
# iter  3 act 1.350e+00 pre 1.159e+00 delta 1.403e+00 f 7.704e+00 |g| 4.734e+00 CG   2
# iter  4 act 1.224e-01 pre 1.142e-01 delta 1.403e+00 f 6.354e+00 |g| 1.152e+00 CG   2
# iter  5 act 2.367e-03 pre 2.344e-03 delta 1.403e+00 f 6.231e+00 |g| 1.366e-01 CG   3
# iter  6 act 1.665e-06 pre 1.665e-06 delta 1.403e+00 f 6.229e+00 |g| 2.961e-03 CG   3
# iter  1 act 1.176e+01 pre 1.071e+01 delta 1.123e+00 f 5.199e+01 |g| 2.337e+01 CG   2
# iter  2 act 1.347e+00 pre 1.275e+00 delta 1.123e+00 f 4.023e+01 |g| 4.788e+00 CG   3
# iter  3 act 5.704e-02 pre 5.669e-02 delta 1.123e+00 f 3.888e+01 |g| 6.592e-01 CG   4
# iter  4 act 1.798e-05 pre 1.797e-05 delta 1.123e+00 f 3.882e+01 |g| 1.694e-02 CG   4
# iter  1 act 2.684e+01 pre 2.335e+01 delta 1.378e+00 f 5.199e+01 |g| 4.560e+01 CG   2
# iter  2 act 6.035e+00 pre 4.936e+00 delta 1.518e+00 f 2.514e+01 |g| 1.197e+01 CG   3
# iter  3 act 1.592e+00 pre 1.327e+00 delta 1.518e+00 f 1.911e+01 |g| 4.287e+00 CG   3
# iter  4 act 1.917e-01 pre 1.813e-01 delta 1.518e+00 f 1.752e+01 |g| 1.165e+00 CG   3
# iter  5 act 2.459e-03 pre 2.451e-03 delta 1.518e+00 f 1.733e+01 |g| 1.219e-01 CG   4
# iter  6 act 6.683e-07 pre 6.682e-07 delta 1.518e+00 f 1.732e+01 |g| 1.946e-03 CG   3
# [LibLinear]Confusion Matrix
#     0   1   2
# 0  23   0   0
# 1   0  24   1
# 2   0   4  23

# Out of 75 Rows, 70 were predicticted correctly
# Which Gives a Prediction Accuracy of 93.33%
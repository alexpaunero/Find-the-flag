import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
print(df.landmass.value_counts())

#Create a new dataframe with only flags from Europe and Oceania
df_36 = df[(df["landmass"].isin([3,6]))]

#Print the average vales of the predictors for Europe and Oceania
print(df.groupby('landmass')[var].mean())

#Create labels for only Europe and Oceania
labels = (df["landmass"].isin([3,6]))*1

#Print the variable types for the predictors
print(df_36[var].dtypes)
print(labels.head())

#Create dummy variables for categorical predictors
data = pd.get_dummies(df[var])
print(data.head())

#Split data into a train and test set
X = data
y = labels
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1, test_size=.4)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []
for i in depths:
  dtc = DecisionTreeClassifier(random_state = 1, max_depth = i)
  dtc.fit(X_train, y_train)
  score = dtc.score(X_test, y_test)
  acc_depth.append(score)
  print(i, score)

#Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.show()

#Find the largest accuracy and the depth this occurs
max_acc = np.max(acc_depth)
print("Max score: " + str(max_acc))
top_depth = acc_depth.index(max_acc) + 1
print("Max depth: " + str(top_depth))

#Refit decision tree model with the highest accuracy and plot the decision tree
dtc = DecisionTreeClassifier(random_state = 1, max_depth = 4)
dtc.fit(X_train, y_train)
tree.plot_tree(dtc, filled = True)
plt.show()

#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

print(ccp_alphas)
acc_pruned = []
for ccp in ccp_alphas:
  dtc = DecisionTreeClassifier(random_state = 1, ccp_alpha = ccp)
  dtc.fit(X_train, y_train)
  score = dtc.score(X_test, y_test)
  acc_pruned.append(score)
  print(ccp, score)

#Plot the accuracy vs ccp_alpha
plt.clf()
plt.plot(ccp_alphas, acc_pruned)
plt.show()

#Find the largest accuracy and the ccp value this occurs
max_score = np.max(acc_pruned)
print("Max score: " + str(max_score))
top_ccp_index = acc_depth.index(max_score)
top_ccp = ccp_alphas[top_ccp_index]
print("Max ccp: " + str(top_ccp))

#Fit a decision tree model with the values for max_depth and ccp_alpha found above
dtc = DecisionTreeClassifier(random_state = 1, max_depth = top_depth, ccp_alpha = top_ccp)
dtc.fit(X_train, y_train)
print(dtc.score(X_test, y_test))

#Plot the final decision tree
tree.plot_tree(dtc, filled = True)
plt.show()

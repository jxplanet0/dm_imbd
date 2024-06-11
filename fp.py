#!/usr/bin/env python
# coding: utf-8

# In[19]:

import graphviz
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[3]:


data = pd.read_csv("kel4_dataset.csv")
data.dropna(inplace=True)  
data['Frontal Axis (G)'] = data['Frontal Axis (G)'].astype(float)
data['Vertical Axis (G)'] = data['Vertical Axis (G)'].astype(float)
data['Lateral Axis (G)'] = data['Lateral Axis (G)'].astype(float)
data['Antenna ID'] = data['Antenna ID'].astype(int)
data['Activity Label'] = data['Activity Label'] - 1
data['Time (s)'] = pd.to_numeric(data['Time (s)'], errors='coerce') 


X = data[['Time (s)', 'Frontal Axis (G)', 'Vertical Axis (G)', 'Lateral Axis (G)', 'Antenna ID', 'RSSI', 'Phase', 'Frequency']]
y = data['Activity Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:
# ...

data.hist(bins=50, figsize=(20, 15))
plt.savefig('histogram.png')


# ...
dataq = data.drop("ID", axis=1)

correlation_matrix = dataq.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.savefig('correlation_matrix.png')


# In[6]:



# In[9]:


params = {
    'learning_rate': 0.1,
    'num_leaves': 30
}

lgb_estimator = lgb.LGBMClassifier(objective='multiclass', num_class=4, metric='multi_error', **params)
lgb_estimator.fit(X_train, y_train)

y_pred_lgb = lgb_estimator.predict(X_test)

accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print("Akurasi model LightGBM dengan parameter:", accuracy_lgb)


# In[10]:


rsme = mean_squared_error(y_test, y_pred_lgb, squared=False)
mae = mean_absolute_error(y_test, y_pred_lgb)
precision = precision_score(y_test, y_pred_lgb, average='macro')
recall = recall_score(y_test, y_pred_lgb, average='macro')
f1 = f1_score(y_test, y_pred_lgb, average='macro')

print("RSME:", rsme)
print("MAE:", mae)
print("Presisi:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[14]:

lgb.plot_tree(lgb_estimator, tree_index=0, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
plt.savefig('tree_plot.png')

# In[16]:

cm = confusion_matrix(y_test, y_pred_lgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lgb_estimator.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')

# In[17]:


lgb.plot_importance(lgb_estimator, max_num_features=10, importance_type='split')
plt.title("Feature Importance - Split")
plt.savefig('feature_importance_split.png')

lgb.plot_importance(lgb_estimator, max_num_features=10, importance_type='gain')
plt.title("Feature Importance - Gain")
plt.savefig('feature_importance_gain.png')

# In[18]:

import numpy as np
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(lgb_estimator, X_train, y_train, cv=5, scoring='accuracy')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

train_std[train_std == 0] = np.zeros_like(train_mean)
val_std[val_std == 0] = np.zeros_like(val_mean)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', label='Cross-validation score')

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)

plt.xlabel('Number of Training Examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.savefig("learning_curve.png")


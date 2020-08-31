#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, roc_auc_score, recall_score, classification_report
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model


# In[2]:


student_data = pd.read_csv('dataset.csv', header = 0)
student_data.head()


# In[3]:


student_data.head()


# In[4]:


print(student_data.shape) 
student_data.dtypes


# In[5]:


target = student_data["Student Outcome"]


# In[6]:


target.head()


# In[7]:


corr = student_data.corr(method='pearson').abs() 
print(corr)


# In[8]:


mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True 
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(16, 10))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)


# In[9]:


plt.figure(figsize=(25,12))
sns.barplot(data=student_data.head(100), x= "Ending Term Unmet Need", y="Cumulated Debt", hue="Student Outcome")


# In[10]:


plt.figure(figsize=(25,12))
sns.barplot(data=student_data.head(100), x= "Cumulative DFW Grades", y="Cumulated Debt", hue="Student Outcome")


# In[11]:


del student_data["Student Outcome"]
del student_data["Transfer Instituion State"]
del student_data["Transfer Instituion Name"]
del student_data["Transfer Region"]
del student_data['Transfer = Home Region Value']
del student_data["Transfer ZIP"]
del student_data["Student Number2"]
del student_data["Lost Tuition Revenue"]
del student_data["Miles from Home"]
del student_data["Ending Term Cumulated GPA"]
del student_data["Ending Term Cumulative GPA"]
del student_data["Transfer Time in Years"]
reg = linear_model.LassoCV()
reg.fit(student_data, target)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(student_data,target)) 
coef = pd.Series(reg.coef_, index = student_data.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")


# In[12]:


imp_coef = coef.sort_values() 
plt.rcParams['figure.figsize'] = (5.0, 3.0)

relevant_features = imp_coef[imp_coef!=0] 
print(relevant_features) 
relevant_features.plot(kind = "barh") 
plt.title("Feature importance using Lasso Model")
 


# In[13]:


data = student_data[[ 'Age',
                     "Starting Term CRHR Completion", 
                     "Pre-Enrollment Credits", 
                     "Ending Term Credits",
                     "Cumulated Debt",
                     'Ending Term Department', 
                     "Ending Term Unmet Need", 
                     "Average Debt Per Term", 
                     "Starting Term Cumulative GPA", 
                     "Ending Term CRHR Completion"
                     ]] 
data.head()


# In[14]:


data.corr(method="pearson")


# In[15]:


data.corr(method="spearman")


# In[16]:


#Logistic Regression
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30, random_state=101)
X_test.shape


# In[17]:


from sklearn.linear_model import LogisticRegression#create an instance and fit the model
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[18]:


#predictions
predictions = logmodel.predict(X_test)


# In[19]:


print(classification_report(y_test,predictions))


# In[20]:


def multiclass_roc_auc_score(y_test, y_pred, average="macro"): 
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


# In[21]:


print("Accuracy", accuracy_score(y_test, predictions)) 
print("ROC", multiclass_roc_auc_score(y_test, predictions))


# In[22]:


import itertools
def plot_confusion_matrix(cm, classes,normalize=False, cmap=plt.cm.Oranges):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        print("Normalized confusion matrix")
        print('Confusion matrix, without normalization')
    else: 
        print(cm)
        
    # Plot the confusion matrix
    plt.figure(figsize = (10, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes)) 
    plt.xticks(tick_marks, classes, rotation=45, size = 14) 
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd' 
    thresh = cm.max() / 2.
        
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size = 18) 
        plt.xlabel('Predicted label', size = 18)


# In[23]:


# Confusion matrix
cm = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cm, classes = ["Dropped Out", "Graduated","Transferred Out" ])


# In[24]:


clf = DecisionTreeClassifier() 

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[25]:


print("Accuracy:",accuracy_score(y_test, y_pred))
print("ROC", multiclass_roc_auc_score(y_test, y_pred))


# In[26]:


print(classification_report(y_test,y_pred))


# In[27]:


from sklearn.tree import export_graphviz 
from sklearn.externals.six import StringIO 
from IPython.display import Image
import pydotplus

dot_data = StringIO() 
export_graphviz(clf, out_file=dot_data, 
                filled=True, 
                rounded=True,
                special_characters=True,feature_names = data.columns) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[28]:


pip install pydotplus


# In[29]:


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3) # Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_prediction = clf.predict(X_test)


# In[30]:


print("Accuracy:",accuracy_score(y_test, y_prediction))
print("ROC", multiclass_roc_auc_score(y_test, y_prediction))


# In[31]:


print(classification_report(y_test,y_prediction))


# In[33]:


dot_data = StringIO() 
export_graphviz(clf, out_file=dot_data,
                filled=True,
                rounded=True,
                special_characters=True,
                feature_names = data.columns,class_names=['0','1','2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())


# In[34]:


cm = confusion_matrix(y_test, y_prediction)
plot_confusion_matrix(cm, classes = ["Dropped Out", "Graduated","Transferred Out" ])


# In[35]:


from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
y_model = model.predict(X_test)


# In[36]:


print("Accuracy", accuracy_score (y_test, y_model)) 
print("ROC", multiclass_roc_auc_score(y_test, y_model))


# In[37]:


# Confusion matrix
cm = confusion_matrix(y_test, y_model)
plot_confusion_matrix(cm, classes = ["Dropped Out", "Graduated","Transferred Out" ])


# In[38]:


print(classification_report(y_test,y_model))


# In[39]:


from sklearn.naive_bayes import GaussianNB 
model = GaussianNB() 
model.fit(X_train, y_train) 
y_p = model.predict(X_test) 


# In[40]:


print("ROC", multiclass_roc_auc_score(y_test, y_p)) 
print("Accuracy", accuracy_score (y_test, y_p))


# In[41]:


# Confusion matrix
cm = confusion_matrix(y_test, y_p)
plot_confusion_matrix(cm, classes = ["Dropped Out", "Graduated","Transferred Out" ])


# In[42]:


print(classification_report(y_test,y_p))


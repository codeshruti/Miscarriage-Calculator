
# coding: utf-8

# The dataset has been created using 9 input features:
# Age : 18-49
# Diabetes : 0 or 1
# Weight : 50 – 80
# Race : 
# 0 – European/American
# 1 – South Asian
# 2- East Asian
# 3 – African or Mixed
# Method of Conception: 
# 0 – Spontaneous
# 1- IVF/Ovulation Drugs
# History of miscarriages: 0 for no, 1 for yes
# Father Age>40: 0 for no, 1 for Yes
# Smoking : 0 for no, 1 for yes
# Drinking : 0 for no, 1 for yes
# 
# 200 attributes are there

# In[247]:


from pandas import read_csv
import pandas as pd
filename = 'miscarriage_calculator.csv'
mc = read_csv(filename)
from sklearn import preprocessing


# In[248]:


mc.head()


# In[249]:


mc.info()


# In[250]:


# Assigning categorical variables
mc['Diabetes'] = mc['Diabetes'].astype('category')
mc['Conception'] = mc['Conception'].astype('category')
mc['PM'] = mc['PM'].astype('category')
mc['Race'] = mc['Race'].astype('category')
mc['FatherAge > 40'] = mc['FatherAge > 40'].astype('category')
mc['Smoking'] = mc['Smoking'].astype('category')
mc['Drinking'] = mc['Drinking'].astype('category')
mc['Miscarriage'] = mc['Miscarriage'].astype('category')


# In[251]:


mc.head()


# In[252]:


# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
mc['Miscarriage']= label_encoder.fit_transform(mc['Miscarriage']) 


# In[253]:


mc['Diabetes']= mc['Diabetes'].cat.codes 
mc['Conception']= mc['Conception'].cat.codes 
mc['PM']= mc['PM'].cat.codes
mc['Race']= mc['Race'].cat.codes 
mc['FatherAge > 40']= mc['FatherAge > 40'].cat.codes
mc['Smoking']= mc['Smoking'].cat.codes 
mc['Drinking']= mc['Drinking'].cat.codes


# In[254]:


mc.head()


# In[255]:


X = mc.drop('Miscarriage',axis=1)
y = mc['Miscarriage']


# In[256]:


mc.head()


# In[257]:


# Dataset is split to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)


# In[258]:


# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# feature extraction
test = SelectKBest(score_func=f_classif, k=5)
fit = test.fit(X_train, y_train)
features_name = test.get_support(indices=True)
# summarize scores
set_printoptions(precision=2)
print(fit.scores_)
print(features_name)
features = fit.transform(X_train)
# summarize selected features , number is the record number
print(features[0:5,:])


# In[259]:


l =[0,1,4,5,6]
col_names = list(X.columns)
for i in l:
    print(col_names[i])


# In[260]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[261]:


X_5f=mc[['Age','Weight','PM','FatherAge > 40','Race']]

# In[263]:


X_5f_train, X_5f_test, y_train, y_test = train_test_split(X_5f, y, test_size=0.30, random_state=40)


# In[282]:


# SVM
from sklearn.svm import SVC
svm_classifier = SVC(decision_function_shape='ovr',kernel='linear',probability=True)
svm_classifier.fit(X_5f_train,y_train)
svm_prediction = svm_classifier.predict(X_5f_test)
print(confusion_matrix(y_test,svm_prediction))
print(accuracy_score(y_test,svm_prediction))
print(classification_report(y_test,svm_prediction))


# In[283]:


# Trying Streamlit
import joblib
with open('model-v1.joblib', 'wb') as f:
    joblib.dump(svm_classifier,f)


# In[284]:


def yes_or_no(value):
    if value == 'Yes':
        return 1
    else:
        return 0


# In[285]:


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
    st.sidebar.subheader('Momma Details')
    Age = st.sidebar.slider('Your Age(in years)',18,49)
    Weight = st.sidebar.slider('Your Weight(in Kg)',45,79)
    PM_cat = st.sidebar.selectbox("Have you had prior miscarriage(s)",('No','Yes'))
    PM = yes_or_no(PM_cat)
    Race_cat  = st.sidebar.selectbox("Select your race",('American/European','Asian','East Asian','Mixed/African'))
    if Race_cat == 'American/European':
        Race = 0
    elif Race_cat == 'Asian':
        Race = 1
    elif Race_cat == 'East Asian':
        Race = 2
    else:
        Race = 3
    FatherAge_cat = st.sidebar.selectbox("Is the Father's Age >40 ",('No','Yes'))
    FatherAge = yes_or_no(FatherAge_cat)
    
    features = {'Age': Age,
            'Weight': Weight,
            'PM': PM,
            'Race': Race,
            'FatherAge > 40': FatherAge
               }
    data = pd.DataFrame(features,index=[0])

    return data


# In[286]:


import streamlit as st
user_input_df = get_user_input()


# In[287]:


def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index=['Low','Medium','High'])
    max_percentage = grad_percentage['Percentage'].max()
    result_index = grad_percentage.idxmax(axis = 0) 
    result = pd.DataFrame(data=max_percentage,columns = ['Risk'],index = result_index)
    if result_index[0] =='Low':
        colour = 'green'
    elif result_index[0] =='Medium':
        colour = 'yellow'
    elif result_index[0] =='High':
        colour = 'red'
    ax = result.plot(kind='barh', figsize=(7, 4),zorder=10, width=0.1,color = colour,visible=True)
    ax.legend().set_visible(True) 
    ax.set_xlim(xmin=0, xmax=100) 
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False) 
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xticks([0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
       ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel("Percentage(%)", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Chance of Miscarriage", labelpad=2, weight='bold', size=12)
    ax.set_title('Risk of Miscarriage ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    st.title(str(max_percentage) + " % : " + result_index[0] +" Risk ")
    return



# In[288]:

st.set_option('deprecation.showPyplotGlobalUse', False)


# In[289]:

prediction_proba = svm_classifier.predict_proba(user_input_df)
visualize_confidence_level(prediction_proba)



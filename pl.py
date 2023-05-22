from scipy import stats
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,SVR 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange
import numpy as np


st.title("hyperparamtres thechnique tunning")


import pandas as pd
inputchoise={"machine_learning_algorithm":"","metric":"","cv":1,"hyperparamtres_thechnique":""}
button_clicked = st.button("Click me")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

# load data
if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.dataframe(data)
    #select Y:

    y=data.columns.tolist()
    select_y=st.selectbox('select Y',y)
    #select x:
    selected_options = st.multiselect('Select X', y)
# Check if the button is clicked
if button_clicked:
    st.write("Button clicked!")
#select 
#select Y:


#select machine learnig model
options_machine_learning = ['random forest classifier ', 'support vector machine', 'K-nearest neighbors','ann','random forest regressor ','support vector regressor ','k neirgbost regressor']
selected_option_m_l_a = st.selectbox('select machine learning model', options_machine_learning)
#select hyperparametres
if selected_option_m_l_a==options_machine_learning[0]:
    st.text("n_estimators")
    n_estimators_min = int(st.text_input("n estimators min", value=0))
    n_estimators_max = int(st.text_input("n estimators max", value=1))
    step_size = int(st.text_input("step size", value=1))
    n_estimators =np.arange(n_estimators_min, n_estimators_max + step_size, step_size)

    st.text("max_depths")
    max_depths_min = int(st.text_input("max_depths min", value=0))
    max_depths_max = int(st.text_input("max_depths max", value=1))
    step_size_max_depths = int(st.text_input("step size f", value=1))
    max_depths = [i * float(step_size) for i in range(int(max_depths_min), int(max_depths_max)+1)]
    st.text("min samples leaf")
    min_samples_leaf_min = int(st.text_input("min_samples_leaf min", value=0))
    min_samples_leaf_max = int(st.text_input("max_depths max ", value=1))
    step_size = int(st.text_input("step size m", value=1))
    min_samples_leaf = [i * int(step_size) for i in range(int(min_samples_leaf_min), int(min_samples_leaf_max)+1)]
    st.text("max features")
    max_features_min = float(st.text_input("max_features min", value="0.1"))
    max_features_max = float(st.text_input("max_features max", value="1.0"))
    step_size = float(st.text_input("step size a", value="0.1"))
    max_features = np.arange(max_features_min, max_features_max + step_size, step_size)
    optin_criteria=['gini','entropy']
    
    hyperparameters = {
    "n_estimators": n_estimators,
   
    "min_samples_leaf": min_samples_leaf,
    
    "criterion": optin_criteria}
    print(hyperparameters)
    
    
    

#print the hyperparametes selected 
st.write( hyperparameters)
#select metric:
options_metric = ['neg_mean_squared_error', 'accuracy']
selected_option_metrcix = st.selectbox('select metric', options_metric)
#cv:
cv_input = st.number_input('Enter nomber of cros folder', step=1, value=1, format='%d')

import streamlit as st

# hyperparametres for tunnuing 




   


    
import streamlit as st



#select hyperparametres thechning :
option_hyperp_thechnique = ['grid search  ', 'random search ', 'hyperband serachcv','bayesion Optimization with gaussian process','bayesion Optimization with bo-tpe','gradient descent ','pso','genetic  algorithms']
selected_option_h_t = st.selectbox('select hyperparametres thechnique ', option_hyperp_thechnique)
#progress bar :
import time
st.text("progression :")
# Create a progress bar




#machine learning models:
class ml_model:
    def LG():
        model=LogisticRegression()
        return model
    def RFc():
        model =  RandomForestClassifier()
        return model
    def svc():
        model = SVC()
        return model
    def knn():
        model=knn = KNeighborsClassifier(n_neighbors=3)
        return model
    def RFG():
        model=rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        return model
    def svg():
        model=SVR()
        return model
    
  
        


#dictionary of user input 

btn_hyperp=st.button('get the hyperpametes')       
from sklearn import datasets
if btn_hyperp:
 inputchoise['machine_learning_algorithm']=selected_option_m_l_a
 inputchoise['hyperparamtres_thechnique']=selected_option_h_t
 inputchoise['metric']=options_metric
 inputchoise['cv']=cv_input  
 rf_params = {
    'n_estimators': [10, 20, 30],'max_depth': [15,20,30,40],"criterion":['gini','entropy']}
    
 print(inputchoise['machine_learning_algorithm'])
 print(inputchoise['hyperparamtres_thechnique'])   
 data = datasets.load_digits()
 X = data.data
 y = data.target
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf = ml_model.RFc()
     grid = GridSearchCV(clf, hyperparameters, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
     
     print("Accuracy:"+ str(grid.best_score_))
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     mod=ml_model.svc()
     rf_params = {
     'C': [1,10, 100],
     "kernel":['poly','rbf','sigmoid'],
     "epsilon":[0.01,0.1,1]}
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf=ml_model.knn()
     rf_params = {
    'n_neighbors': [2, 4, 5, 7, 10]}
     
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf=ml_model.RFG()
     rf_params = {
    'n_estimators': [10, 20, 12],'max_depth': [15,20,30,50]}
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     rf_params = {'C': [1,10, 100],"kernel":['poly','rbf','sigmoid'],"epsilon":[0.01,0.1,1]}
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, rf_params, cv=int(cv_input), scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
      rf_params = {
    'n_estimators': sp_randint(10,100),
    "max_features":sp_randint(1,64),
    'max_depth': sp_randint(5,50),
    "min_samples_split":sp_randint(2,11),
    "min_samples_leaf":sp_randint(1,11),
    "criterion":['gini','entropy']}
      n_iter_search=20 #number of iterations is set to 20, you can increase this number if time permits
      clf = RandomForestClassifier(random_state=0)
      Random = RandomizedSearchCV(clf, param_distributions=hyperparameters,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
      Random.fit(X, y)
      print(Random.best_params_)
      print("Accuracy:"+ str(Random.best_score_))  
      df_best_params = pd.DataFrame(Random.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
     rf_params = {'C': stats.uniform(0,50),"kernel":['linear','poly','rbf','sigmoid']}
     n_iter_search=20
     clf = SVC(gamma='scale')
     Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
     Random.fit(X, y)
     print(Random.best_params_)
     print("Accuracy:"+ str(Random.best_score_))
     df_best_params = pd.DataFrame(Random.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
     rf_params = {'C': stats.uniform(0,50),"kernel":['linear','poly','rbf','sigmoid']}
     n_iter_search=20
     clf = SVC(gamma='scale')
     Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
     Random.fit(X, y)
     print(Random.best_params_)
     print("Accuracy:"+ str(Random.best_score_)) 
     df_best_params = pd.DataFrame(Random.best_params_, index=[0])  
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
     rf_params = {'n_neighbors': range(1,20),}
     n_iter_search=10
     clf = KNeighborsClassifier()
     Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
     Random.fit(X, y)
     print(Random.best_params_)
     print("Accuracy:"+ str(Random.best_score_)) 
     df_best_params = pd.DataFrame(Random.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
     rf_params = {'n_estimators': sp_randint(10,100),"max_features":sp_randint(1,13),'max_depth': sp_randint(5,50),"min_samples_split":sp_randint(2,11),"min_samples_leaf":sp_randint(1,11),"criterion":['mse','mae']}
     n_iter_search=20 #number of iterations is set to 20, you can increase this number if time permits
     clf = RandomForestRegressor(random_state=0)
     Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
     Random.fit(X, y)

     print(Random.best_params_)
     print("Accuracy:"+ str(Random.best_score_)) 
     df_best_params = pd.DataFrame(Random.best_params_, index=[0])                        
 
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     clf = ml_model.RFc()
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
     
     print("Accuracy:"+ str(grid.best_score_))
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     mod=ml_model.svc()
     rf_params = {
     'C': [1,10, 100],
     "kernel":['poly','rbf','sigmoid'],
     "epsilon":[0.01,0.1,1]}
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     clf=ml_model.knn()
     rf_params = {
     'n_neighbors': [2, 4, 5, 7, 10]}
     
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     clf=ml_model.RFG()
     rf_params = {
     'n_estimators': [10, 20, 12],'max_depth': [15,20,30,50]}
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     print("kd")
 
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
     clf = ml_model.RFc()
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
     
     print("Accuracy:"+ str(grid.best_score_))
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
     mod=ml_model.svc()
     rf_params = {
     'C': [1,10, 100],
     "kernel":['poly','rbf','sigmoid'],
     "epsilon":[0.01,0.1,1]}
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
     clf=ml_model.knn()
     rf_params = {
     'n_neighbors': [2, 4, 5, 7, 10]}
     
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
     clf=ml_model.RFG()
     rf_params = {
     'n_estimators': [10, 20, 12],'max_depth': [15,20,30,50]}
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
     print("ke")
 
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     clf = ml_model.RFc()
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
     
     print("Accuracy:"+ str(grid.best_score_))
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     mod=ml_model.svc()
     rf_params = {
     'C': [1,10, 100],
     "kernel":['poly','rbf','sigmoid'],
     "epsilon":[0.01,0.1,1]}
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     clf=ml_model.knn()
     rf_params = {
     'n_neighbors': [2, 4, 5, 7, 10]}
     
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     clf=ml_model.RFG()
     rf_params = {
     'n_estimators': [10, 20, 12],'max_depth': [15,20,30,50]}
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     print("ke")
     
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     clf = ml_model.RFc()
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
     
     print("Accuracy:"+ str(grid.best_score_))
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     mod=ml_model.svc()
     rf_params = {
     'C': [1,10, 100],
     "kernel":['poly','rbf','sigmoid'],
     "epsilon":[0.01,0.1,1]}
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     clf=ml_model.knn()
     rf_params = {
     'n_neighbors': [2, 4, 5, 7, 10]}
     
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     clf=ml_model.RFG()
     rf_params = {
     'n_estimators': [10, 20, 12],'max_depth': [15,20,30,50]}
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     print("jd")
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:
     clf = ml_model.RFc()
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
     
     print("Accuracy:"+ str(grid.best_score_))
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:
     mod=ml_model.svc()
     rf_params = {
     'C': [1,10, 100],
     "kernel":['poly','rbf','sigmoid'],
     "epsilon":[0.01,0.1,1]}
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:
     clf=ml_model.knn()
     rf_params = {
     'n_neighbors': [2, 4, 5, 7, 10]}
     
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:
     clf=ml_model.RFG()
     rf_params = {
     'n_estimators': [10, 20, 12],'max_depth': [15,20,30,50]}
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:    
     print("kdje")
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[7]:
     clf = ml_model.RFc()
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
     
     print("Accuracy:"+ str(grid.best_score_))
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[7]:
     mod=ml_model.svc()
     rf_params = {
     'C': [1,10, 100],
     "kernel":['poly','rbf','sigmoid'],
     "epsilon":[0.01,0.1,1]}
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[7]:
     clf=ml_model.knn()
     rf_params = {
     'n_neighbors': [2, 4, 5, 7, 10]}
     
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[7]:
     clf=ml_model.RFG()
     rf_params = {
     'n_estimators': [10, 20, 12],'max_depth': [15,20,30,50]}
     grid = GridSearchCV(clf, rf_params, cv=cv_input, scoring=str(selected_option_metrcix))
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[7]:    
     print("kde")
 else:
     print("js")     
 st.title("the hyperparametes is :")
 st.table(df_best_params) 
 st.title("Accuracy:"+ str(grid.best_score_))
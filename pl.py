from scipy import stats
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,SVR 
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange
import numpy as np
from hyperband import HyperbandSearchCV
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import optunity
import optunity.metrics
import time
from skopt import Optimizer
from skopt import BayesSearchCV 
from skopt.space import Real, Categorical, Integer
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
st.set_page_config(page_title='The Machine Learning Hyperparameter Optimization App',
    layout='wide')
st.write("""
# The Machine Learning Hyperparameter Optimization App """)
total_time=0
df_best_score=0

import pandas as pd
inputchoise={"machine_learning_algorithm":"","metric":"","cv":1,"hyperparamtres_thechnique":""}

st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# load data
if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Display the DataFrame
    
    st.subheader('DATA')
    st.write(data)
    st.subheader('DATA INFORMATION')
    st.write(data.describe())
  
    
    #select Y:
    
    y=data.columns.tolist()
    select_y=st.sidebar.selectbox('select Y',y,index=y.index(y[-1]))
    #select x:
    select_x_options = [option for option in y if option != select_y]
    select_all_x = st.sidebar.checkbox('Select all X')
    if select_all_x:
     selecte_x = st.sidebar.multiselect('Select X', select_x_options, default=select_x_options)
    else:
      selecte_x = st.sidebar.multiselect('Select X', select_x_options)

   
    X=data[selecte_x]
    y=data[select_y]
    X_table = pd.DataFrame(X, index=[0])
    Y_table = pd.DataFrame(y, index=[0])
    st.subheader("X selected")
    st.table(X_table)
    st.subheader("Y selected")
    st.table(Y_table)
   
# Check if the button is clicked
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The **Diabetes** dataset is used as the example.')
        st.write(df.head(5))
#select X:

#select Y:


#select machine learnig model
options_machine_learning = ['random forest classifier ', 'support vector machine', 'K-nearest neighbors','random forest regressor ','support vector regressor ','k neirgbost regressor']
selected_option_m_l_a = st.sidebar.selectbox('select machine learning model', options_machine_learning)
st.subheader("The machine learning algorithme is :")
st.write(selected_option_m_l_a)
#select hyperparametres
if selected_option_m_l_a==options_machine_learning[0]:
    st.sidebar.text("n_estimators")
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 1, 100, (1,10), 1)
    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 1)
    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
    st.sidebar.text("max_depths")
    parameter_max_depths = st.sidebar.slider('Number of max depths (max_depths)', 0, 100, (1,2), 1)
    parameter_max_depths_step = st.sidebar.number_input('Step size for max_depths', 1)
    max_depths_range = np.arange(parameter_max_depths[0], parameter_max_depths[1]+parameter_n_estimators_step, parameter_max_depths_step)
    st.sidebar.text("min samples leaf")
    parameter_min_samples = st.sidebar.slider('Number of min samples leaf (min_samples_leaf)', 0, 100, (10,50), 50)
    parameter_min_samples_leaf_step = st.sidebar.number_input('Step size for min_samples_leaf', 10)
    min_samples_leaf_range = np.arange(parameter_max_depths[0], parameter_max_depths[1]+parameter_n_estimators_step, parameter_max_depths_step)
    st.sidebar.text("max features")
    parameter_max_features = st.sidebar.slider('Number of max features (max_features)', 0, 100, (1,10), 1)
    parameter_max_features_step = st.sidebar.number_input('Step size for max_features', 1)
    max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+parameter_max_features_step, parameter_max_features_step)
    optin_criteria=['gini','entropy']
    
    hyperparameters = {
    "n_estimators": n_estimators_range,
    "max_features" : max_features_range,
    'max_depth' : max_depths_range,
    "min_samples_leaf": min_samples_leaf_range,
    "criterion": optin_criteria}
    
    
    
if selected_option_m_l_a==options_machine_learning[1]:
    st.sidebar.text("C")
    parameter_c = st.sidebar.slider('Number of C (C)', 0, 100, (1,5), 1)
    parameter_c_step = st.sidebar.number_input('Step size for c', 1)
    c_range = np.arange(parameter_c[0], parameter_c[1]+parameter_c_step, parameter_c_step)
    st.sidebar.text("Kernel")
    options_kernel = ['linear', 'poly', 'rbf','sigmoid']
    kernel = st.sidebar.multiselect('select kernel', options_kernel)
    hyperparameters = {
    "C": c_range,
   
    "kernel": kernel,
    
    }
    
        
if selected_option_m_l_a==options_machine_learning[2]:
    st.sidebar.text("n_neighbors")
    parameter_n_neighbors = st.sidebar.slider('Number of neighbors (n_neighbors)', 0, 100, (1,10), 1)
    parameter_n_neighbors_step = st.sidebar.number_input('Step size for n_neighbors', 1)
    n_neighbors_range = np.arange(parameter_n_neighbors[0], parameter_n_neighbors[1]+parameter_n_neighbors_step, parameter_n_neighbors_step)
    st.sidebar.text("weight")
    options_weights = ['uniform', 'distance']
    weights = st.sidebar.multiselect('select weight', options_weights)
    st.sidebar.text("metric")
    options_metric = ['minkowski', 'euclidean','manhattan']
    metrics = st.sidebar.multiselect('select metric', options_metric)
    hyperparameters = {
    "n_neighbors": n_neighbors_range,
    'weights' : weights,
    'metric' : metrics
    }
    

if selected_option_m_l_a==options_machine_learning[3]:
    st.sidebar.text("n_estimators")
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 100, (10,50), 1)
    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
    st.sidebar.text("max_depths")
    parameter_max_depths = st.sidebar.slider('Number of max depths (max_depths)', 0, 100, (1,5), 1)
    parameter_max_depths_step = st.sidebar.number_input('Step size for max_depths', 1)
    max_depths_range = np.arange(parameter_max_depths[0], parameter_max_depths[1]+parameter_max_depths_step, parameter_max_depths_step)
    st.sidebar.text("min samples leaf")
    parameter_min_samples = st.sidebar.slider('Number of min samples leaf (min_samples_leaf)', 0, 100, (10,50), 1)
    parameter_min_samples_leaf_step = st.sidebar.number_input('Step size for min_samples_leaf', 1)
    min_samples_leaf_range = np.arange(parameter_min_samples[0], parameter_min_samples[1]+parameter_min_samples_leaf_step, parameter_min_samples_leaf_step)
    st.sidebar.text("max features")
    parameter_max_features = st.sidebar.slider('Number of max features (max_features)', 0, 100, (0,10), 1)
    parameter_max_features_step = st.sidebar.number_input('Step size for max_features', 1)
    max_features_range = np.arange(parameter_max_features [0], parameter_max_features [1]+parameter_max_features_step, parameter_max_features_step)
    optin_criteria=['mse','mae']
    hyperparameters = {
    "n_estimators": n_estimators_range,
    "max_features" : max_features_range,
    'max_depth' : max_depths_range,
    "min_samples_leaf": min_samples_leaf_range,
    "criterion": optin_criteria}
    

if selected_option_m_l_a==options_machine_learning[4]:
    st.sidebar.text("C")
    parameter_c = st.sidebar.slider('Number of C (C)', 0, 100, (10,50), 1)
    parameter_c_step = st.sidebar.number_input('Step size for c', 1)
    c_range = np.arange(parameter_c[0], parameter_c[1]+parameter_c_step, parameter_c_step)
    
    options_kernel = ['linear', 'poly', 'rbf','sigmoid']
    kernel = st.sidebar.multiselect('select kernel', options_kernel)
    hyperparameters = {
    "C": c_range,
    "kernel": kernel
   
    }
    
if selected_option_m_l_a==options_machine_learning[5]:
    st.sidebar.text("n_neighbors")
    parameter_n_neighbors = st.sidebar.slider('Number of neighbors (n_neighbors)', 0, 100, (1,5), 1)
    parameter_n_neighbors_step = st.sidebar.number_input('Step size for n_neighbors', 1)
    n_neighbors_range = np.arange(parameter_n_neighbors[0], parameter_n_neighbors[1]+parameter_n_neighbors_step, parameter_n_neighbors_step)
    st.sidebar.text("weight")
    options_weights = ['uniform', 'distance']
    weights = st.sidebar.multiselect('select weight', options_weights)
    st.sidebar.text("metric")
    options_metric = ['minkowski', 'euclidean','manhattan']
    metrics = st.sidebar.multiselect('select metric', options_metric)
    hyperparameters = {
    "n_neighbors": n_neighbors_range,
    'weights' : weights,
    'metric' : metrics
    }
#print the hyperparametes selected 

st.subheader("the hyperparameters is :")
st.write( hyperparameters)
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 100, 80, 1)
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=split_size)

#select metric:
options_metric = ['neg_mean_squared_error', 'accuracy']
selected_option_metrcix = st.sidebar.selectbox('select metric', options_metric)
st.subheader("the Metric is : ")
st.write(selected_option_metrcix )

#cv:
cv_input = st.sidebar.number_input('Enter nomber of cros folder', step=1, value=2, format='%d')
st.subheader("The number of cross-validations : ")
st.write(str(cv_input) )

import streamlit as st





#select hyperparametres thechning :
option_hyperp_thechnique = ['grid search  ', 'random search ', 'hyperband serachcv','bayesion Optimization with gaussian process','bayesion Optimization with bo-tpe',' Particle Swarm Optimization(PSO)','genetic  algorithms']
selected_option_h_t = st.sidebar.selectbox('select hyperparametres thechnique ', option_hyperp_thechnique)
st.subheader("The hyperparametres thechnique that selected is ")
st.write(selected_option_h_t)
if selected_option_h_t == option_hyperp_thechnique[1]:
     n_iter_search = st.sidebar.slider('Number of iterations  (n_iter_search )', 1, 1000, 20, 1)
     
if selected_option_h_t == option_hyperp_thechnique[2]:
     min_iter = st.sidebar.slider('Number of minimum iterations  (min_iter)', 1, 1000, 10, 1)
     max_iter = st.sidebar.slider('Number of maximum  iterations  (max_iter )', 1, 1000, 100, 1)
     
if selected_option_h_t == option_hyperp_thechnique[3]:
     n_iter_search = st.sidebar.slider('Number of iterations  (n_iter_search )', 1, 1000, 20, 1)

if selected_option_h_t == option_hyperp_thechnique[4]:
    max_evals = st.sidebar.slider('the maximum number of evaluations or iterations  (max_evals)', 1, 1000, 20, 1)

if selected_option_h_t == option_hyperp_thechnique[6]:
     population_size = st.sidebar.slider('Population size  (population_size )', 0, 1000, 10, 1)
     gene_mutation_prob = st.sidebar.slider('Gene mutation probability (gene_mutation_prob)', 0.0, 1.0, 0.5, 0.1)
     gene_crossover_prob = st.sidebar.slider('Gene crossover probability (gene_crossover_prob)', 0.0, 1.0, 0.5, 0.1)
     tournament_size = st.sidebar.slider('tournament size (tournament_size)', 0, 1000, 3, 1)
     generations_number = st.sidebar.slider('Generations number (generations_number)', 0, 1000, 5, 1)
st.sidebar.subheader('General Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

# Create a progress bar

   
  
        


#dictionary of user input 

btn_hyperp=st.button('get the hyperpametes')       
from sklearn import datasets
if btn_hyperp:
 inputchoise['machine_learning_algorithm']=selected_option_m_l_a
 inputchoise['hyperparamtres_thechnique']=selected_option_h_t
 inputchoise['metric']=options_metric
 inputchoise['cv']=cv_input  
 X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=split_size)
 X=X_train
 y=Y_train
 

    
   
 
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf = RandomForestClassifier()
     grid = GridSearchCV(clf, hyperparameters, cv=cv_input, scoring=str(selected_option_metrcix))
     start_time=time.time()
     grid.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=grid.best_score_
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])

 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, hyperparameters, cv=cv_input, scoring=str(selected_option_metrcix))
     start_time=time.time()
     grid.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=grid.best_score_
     
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf=KNeighborsClassifier()
     grid = GridSearchCV(clf, hyperparameters, cv=cv_input, scoring=str(selected_option_metrcix))
     start_time=time.time()
     grid.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=grid.best_score_
     
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[3] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf=RandomForestRegressor()
     grid = GridSearchCV(clf, hyperparameters, cv=cv_input, scoring=str(selected_option_metrcix))
     start_time=time.time()
     grid.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=grid.best_score_
     
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, hyperparameters, cv=int(cv_input), scoring=str(selected_option_metrcix))
     start_time=time.time()
     grid.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=grid.best_score_
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf = KNeighborsRegressor()
     grid = GridSearchCV(clf, hyperparameters, cv=int(cv_input), scoring=str(selected_option_metrcix))
     start_time=time.time()
     grid.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=grid.best_score_
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])        
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
      #number of iterations is set to 20, you can increase this number if time permits
      clf = RandomForestClassifier(random_state=parameter_random_state)
      Random = RandomizedSearchCV(clf, param_distributions=hyperparameters,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
      start_time=time.time()
      Random.fit(X, y)
      end_time=time.time()
      total_time=end_time-start_time
      df_best_score=Random.best_score_
      df_best_params = pd.DataFrame(Random.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
     clf = SVC(gamma='scale')
     Random = RandomizedSearchCV(clf, param_distributions=hyperparameters,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
     start_time=time.time()
     Random.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=Random.best_score_
     df_best_params = pd.DataFrame(Random.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
     clf = KNeighborsClassifier()
     Random = RandomizedSearchCV(clf, param_distributions=hyperparameters,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
     Random.fit(X, y) 
     df_best_score=Random.best_score_
     
     df_best_params = pd.DataFrame(Random.best_params_, index=[0])  
 if inputchoise['machine_learning_algorithm']== options_machine_learning[3] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
     clf = RandomForestRegressor()
     Random = RandomizedSearchCV(clf, param_distributions=hyperparameters,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
     start_time=time.time()
     Random.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=Random.best_score_
     
     df_best_params = pd.DataFrame(Random.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
     #number of iterations is set to 20, you can increase this number if time permits
     clf = SVR()
     Random = RandomizedSearchCV(clf, param_distributions=hyperparameters,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
     start_time=time.time()
     Random.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=Random.best_score_
     df_best_params = pd.DataFrame(Random.best_params_, index=[0])                        

 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[1]:
     clf = KNeighborsRegressor()
     Random = RandomizedSearchCV(clf, param_distributions=hyperparameters,n_iter=n_iter_search,cv=int(cv_input),scoring=str(selected_option_metrcix))
     start_time=time.time()
     Random.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=Random.best_score_
     
     df_best_params = pd.DataFrame(Random.best_params_, index=[0])        


 
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     clf = RandomForestClassifier(random_state=parameter_random_state)
     rf_params = {
    'n_estimators': sp_randint(parameter_n_estimators[0],parameter_n_estimators[1]),
    "max_features":sp_randint(parameter_max_features[0],parameter_max_features[1]),
    'max_depth': sp_randint(parameter_max_depths[0],parameter_max_depths[1]),
    "min_samples_split":sp_randint(2,11),
    "min_samples_leaf":sp_randint(parameter_min_samples[0],parameter_min_samples[1]),
    "criterion":['gini','entropy']}
     hyper = HyperbandSearchCV(clf, param_distributions =rf_params,cv=int(cv_input),min_iter=min_iter,max_iter=max_iter,scoring=str(selected_option_metrcix))
     
     start_time=time.time()
     hyper.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=Random.best_score_
     
     df_best_params = pd.DataFrame(hyper.best_params_, index=[0])
     
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     clf = SVC(gamma='scale')
     rf_params = {
     'C': stats.uniform(parameter_c[0],parameter_c[1]),
     "kernel":hyperparameters["kernel"]}
     hyper = HyperbandSearchCV(clf, param_distributions =rf_params,cv=int(cv_input),min_iter=1,max_iter=10,scoring=str(selected_option_metrcix),resource_param='C')
     start_time=time.time()
     hyper.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=hyper.best_score_
     
     df_best_params = pd.DataFrame(hyper.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     clf=KNeighborsClassifier()
     hyper = HyperbandSearchCV(clf, param_distributions =hyperparameters,cv=3,min_iter=1,max_iter=20,scoring=str(selected_option_metrcix),resource_param='n_neighbors')
     start_time=time.time()
     hyper.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=hyper.best_score_
     
     df_best_params = pd.DataFrame(hyper.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[3] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     clf = RandomForestRegressor(random_state=parameter_random_state)
     hyper = HyperbandSearchCV(clf, param_distributions =hyperparameters,cv=3,min_iter=10,max_iter=100,scoring=str(selected_option_metrcix))
     start_time=time.time()
     hyper.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=hyper.best_score_
     
     df_best_params = pd.DataFrame(hyper.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     clf = SVR(gamma='scale')
     hyper = HyperbandSearchCV(clf, param_distributions =hyperparameters,cv=3,min_iter=1,max_iter=10,scoring=str(selected_option_metrcix),resource_param='C')
     start_time=time.time()
     hyper.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=hyper.best_score_
     
     df_best_params = pd.DataFrame(hyper.best_params_, index=[0])    
 
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[2]:
     clf = KNeighborsRegressor()
     hyper = HyperbandSearchCV(clf, param_distributions =hyperparameters,cv=int(cv_input),min_iter=1,max_iter=20,scoring=str(selected_option_metrcix),resource_param='n_neighbors')
     start_time=time.time()
     hyper.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     
     df_best_score=hyper.best_score_
     
     df_best_params = pd.DataFrame(hyper.best_params_, index=[0]) 
       
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
     clf = RandomForestClassifier(random_state=parameter_random_state)
     Bayes = BayesSearchCV(clf, hyperparameters,cv=int(cv_input),n_iter=20, n_jobs=parameter_n_jobs,scoring=str(selected_option_metrcix))
     #number of iterations is set to 20, you can increase this number if time permits
     start_time=time.time()
     Bayes.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=Bayes.best_score_
     
     df_best_params=pd.DataFrame(Bayes.best_params_, index=[0])
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
     clf = SVC(gamma='scale')
     Bayes = BayesSearchCV(clf, hyperparameters,cv=int(cv_input),n_iter=20, n_jobs=parameter_n_jobs,scoring=str(selected_option_metrcix))
     start_time=time.time()
     Bayes.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=Bayes.best_score_
     
     df_best_params = pd.DataFrame(Bayes.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
     clf = KNeighborsClassifier()
     Bayes = BayesSearchCV(clf, hyperparameters,cv=int(cv_input),n_iter=10, n_jobs=parameter_n_jobs,scoring=str(selected_option_metrcix))
     Bayes.fit(X, y)
     df_best_score=Bayes.best_score_
     
     df_best_params = pd.DataFrame(Bayes.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[3] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
     clf = RandomForestRegressor(random_state=parameter_random_state)
     Bayes = BayesSearchCV(clf, hyperparameters,cv=int(cv_input),n_iter=20, scoring=str(selected_option_metrcix)) 
     start_time=time.time()
     Bayes.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=Bayes.best_score_
     
    
     df_best_params = pd.DataFrame(Bayes.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
      clf = SVR(gamma='scale')
      Bayes = BayesSearchCV(clf, hyperparameters,cv=int(cv_input),n_iter=20, scoring=str(selected_option_metrcix))
      start_time=time.time()
      Bayes.fit(X, y)
      end_time=time.time()
      total_time=end_time-start_time
      df_best_score=Bayes.best_score_
      
      print(Bayes.best_params_)
      df_best_params = pd.DataFrame(Bayes.best_params_, index=[0])    
      
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[3]:
      clf = KNeighborsRegressor()
      Bayes = BayesSearchCV(clf, hyperparameters,cv=int(cv_input),n_iter=10, scoring=str(selected_option_metrcix))
      start_time=time.time()
      Bayes.fit(X, y)
      end_time=time.time()
      total_time=end_time-start_time
      df_best_score=Bayes.best_score_
      
      df_best_params = pd.DataFrame(Bayes.best_params_, index=[0])     
      
      
      
      
      
      
      
      
      
      
 
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     def objective(params):
       params = {
        'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth']),
        'max_features': int(params['max_features']),
        "min_samples_split":int(params['min_samples_split']),
        "min_samples_leaf":int(params['min_samples_leaf']),
        "criterion":str(params['criterion'])}
       clf = RandomForestClassifier( **params)
       score = cross_val_score(clf, X, y, scoring=str(selected_option_metrcix), cv=StratifiedKFold(n_splits=int(cv_input))).mean()
       return {'loss':-score, 'status': STATUS_OK }
 # Define the hyperparameter configuration space
     space = {'n_estimators': hp.quniform('n_estimators', parameter_n_estimators[0], parameter_n_estimators[1], parameter_n_estimators_step),
    'max_depth': hp.quniform('max_depth', parameter_max_depths[0], parameter_max_depths[1], parameter_max_depths_step),
    "max_features":hp.quniform('max_features', parameter_max_features[0], parameter_max_features[1], parameter_max_features_step),
    "min_samples_split":hp.quniform('min_samples_split',2,11,1),
    "min_samples_leaf":hp.quniform('min_samples_leaf',parameter_min_samples[0],parameter_min_samples[1],parameter_min_samples_leaf_step),
    "criterion":hp.choice('criterion',['gini','entropy'])}
     start_time=time.time()
     best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=max_evals)
     end_time=time.time()
     total_time=end_time-start_time

     df_best_params = pd.DataFrame(best, index=[0])
     

 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
  def objective(params):
       params = {
        'C': abs(float(params['C'])), 
        "kernel":str(params['kernel'])}
       
       clf = SVC(gamma='scale', **params)
       score = cross_val_score(clf, X, y, scoring=str(selected_option_metrcix), cv=StratifiedKFold(n_splits=cv_input)).mean()
       return {'loss':-score, 'status': STATUS_OK }
 # Define the hyperparameter configuration space
  space = {
      'C': hp.normal('C', parameter_c[0],parameter_c[1]),
      "kernel":hp.choice('kernel',hyperparameters["kernel"])
  }
  start_time=time.time()
  best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=max_evals)
  end_time=time.time()
  total_time=end_time-start_time

  df_best_params = pd.DataFrame(best, index=[0])
     
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     def objective(params):
       params = {'n_neighbors': abs(int(params['n_neighbors'])) }
       clf = KNeighborsClassifier( **params)
       score = cross_val_score(clf, X, y, scoring=str(selected_option_metrcix), cv=StratifiedKFold(n_splits=cv_input)).mean()
       return {'loss':-score, 'status': STATUS_OK }

     space = {'n_neighbors': hp.quniform('n_neighbors', parameter_n_neighbors[0], parameter_n_neighbors[1], parameter_n_neighbors_step)}
     start_time=time.time()
     best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=max_evals)
     end_time=time.time()
     total_time=end_time-start_time
     
     
     df_best_params = pd.DataFrame(best, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[3] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     def objective(params):
       params = { 
        'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth']),
        'max_features': int(params['max_features']),
        "min_samples_split":int(params['min_samples_split']),
        "min_samples_leaf":int(params['min_samples_leaf']),
        "criterion":str(params['criterion'])}
       clf = RandomForestRegressor( **params)
       score = cross_val_score(clf, X, y, scoring=str(selected_option_metrcix), cv=StratifiedKFold(n_splits=int(cv_input))).mean()
       return {'loss':-score, 'status': STATUS_OK }
 # Define the hyperparameter configuration space
     space = {'n_estimators': hp.quniform('n_estimators', parameter_n_estimators[0], parameter_n_estimators[1], parameter_n_estimators_step),
    'max_depth': hp.quniform('max_depth', parameter_max_depths[0], parameter_max_depths[1], parameter_max_depths_step),
    "max_features":hp.quniform('max_features', parameter_max_features[0], parameter_max_features[1], parameter_max_features_step),
    "min_samples_split":hp.quniform('min_samples_split',2,11,1),
    "min_samples_leaf":hp.quniform('min_samples_leaf',parameter_min_samples[0],parameter_min_samples[1],parameter_min_samples_leaf_step),
    "criterion":hp.choice('criterion',['gini','entropy'])}
     start_time=time.time()
     best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=max_evals)
     end_time=time.time()
     total_time=end_time-start_time

     df_best_params = pd.DataFrame(best, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     def objective(params):
       params = {
        'C': abs(float(params['C'])), 
        "kernel":str(params['kernel'])} 
       clf = SVR(gamma='scale', **params)
       score = cross_val_score(clf, X, y, scoring=str(selected_option_metrcix), cv=StratifiedKFold(n_splits=cv_input)).mean()
       return {'loss':-score, 'status': STATUS_OK }
 # Define the hyperparameter configuration space
     space = {'C': hp.normal('C', parameter_c[0],parameter_c[1]),
      "kernel":hp.choice('kernel',hyperparameters["kernel"])}
     start_time=time.time()
     best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=max_evals)
     end_time=time.time()
     total_time=end_time-start_time

     df_best_params = pd.DataFrame(best, index=[0])
 
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[4]:
     def objective(params):
       params = { 'n_neighbors': abs(int(params['n_neighbors'])) }
       clf = KNeighborsRegressor( **params)
       score = cross_val_score(clf, X, y, scoring=str(selected_option_metrcix), cv=StratifiedKFold(n_splits=cv_input)).mean()
       return {'loss':-score, 'status': STATUS_OK }

     space = {'n_neighbors': hp.quniform('n_neighbors', parameter_n_neighbors[0], parameter_n_neighbors[1], parameter_n_neighbors_step)}
     start_time=time.time()
     best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=max_evals)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_params = pd.DataFrame(best, index=[0])    
  
  
  
  
     
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     search = {'n_estimators': [parameter_n_estimators[0], parameter_n_estimators[1]],
    'max_features': [parameter_max_features[0], parameter_max_features[1]],
    'max_depth': [parameter_max_depths[0],parameter_max_depths[1]],
    "min_samples_split":[2,11],
    "min_samples_leaf":[parameter_min_samples[0],parameter_min_samples[1]],
    "criterion":[0,1] }
     d = datasets.load_digits()
     X = d.data
     y = d.target
     data=X
     labels=y.tolist()
     @optunity.cross_validated(x=data, y=labels, num_folds=3)
     def performance(x_train, y_train, x_test, y_test,n_estimators=None, max_features=None,max_depth=None,min_samples_split=None,min_samples_leaf=None,criterion=None):
         if criterion<0.5:
           cri='gini'
         else:
           cri='entropy'
         model = RandomForestClassifier(n_estimators=int(n_estimators),
                                   max_features=int(max_features),
                                   max_depth=int(max_depth),
                                   min_samples_split=int(min_samples_split),
                                   min_samples_leaf=int(min_samples_leaf),
                                   criterion=cri,
                                  )
   
         scores=np.mean(cross_val_score(model, X, y, cv=3, n_jobs=parameter_n_jobs,
                                    scoring=str(selected_option_metrcix)))
    
         return scores

     optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

     df_best_params = pd.DataFrame(optimal_configuration, index=[0])
     
    
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     search = { 'C': (parameter_c[0],parameter_c[1]),'kernel':[0,4]}
     @optunity.cross_validated(x=X, y=y, num_folds=int(cv_input))
      
     def performance(x_train, y_train, x_test, y_test,C=None,kernel=None):
      if kernel<1:
        ke='linear'
      elif kernel<2:
        ke='poly'
      elif kernel<3:
        ke='rbf'
      else:
        ke='sigmoid'   
      model = SVC(C=float(c_range),
                kernel=ke)
   
      scores=np.mean(cross_val_score(model, X, y, cv=int(cv_input), n_jobs=parameter_n_jobs,scoring=str(selected_option_metrcix)))
    
      return scores

     optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

     df_best_params = pd.DataFrame(optimal_configuration, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     search = {'n_neighbors': [parameter_n_neighbors[0], parameter_n_neighbors[1]] }
     @optunity.cross_validated(x=X, y=y, num_folds=int(cv_input))
     def performance(x_train, y_train, x_test, y_test,n_neighbors=None):
          model = KNeighborsClassifier(n_neighbors=int(n_neighbors))   
          scores=np.mean(cross_val_score(model, X, y, cv=int(cv_input), n_jobs=parameter_n_jobs,scoring=str(selected_option_metrcix)))
    
          return scores

     optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

     df_best_params = pd.DataFrame(optimal_configuration, index=[0])  
 if inputchoise['machine_learning_algorithm']== options_machine_learning[3] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     search = { }

     @optunity.cross_validated(x=X, y=y, num_folds=int(cv_input))
     def performance(x_train, y_train, x_test, y_test,n_estimators=None, max_features=None,max_depth=None,min_samples_split=None,min_samples_leaf=None,criterion=None):
      model = KNeighborsClassifier(n_neighbors=int(n_neighbors_range))
             
      scores=np.mean(cross_val_score(model, X, y, cv=int(cv_input), n_jobs=parameter_n_jobs,scoring=str(selected_option_metrcix)))
    
      return scores

     optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

     df_best_params = pd.DataFrame(optimal_configuration, index=[0])     
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[5]:
     print("jd")
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:
      rf_params = {}
      clf = RandomForestClassifier(random_state=parameter_random_state)

      ga1 = EvolutionaryAlgorithmSearchCV(estimator=clf,
                                   params=hyperparameters,
                                   scoring=str(selected_option_metrcix),
                                   cv=int(cv_input),
                                   verbose=1,
                                   population_size=population_size,
                                   gene_mutation_prob=gene_mutation_prob,
                                   gene_crossover_prob=gene_crossover_prob,
                                   tournament_size=tournament_size,
                                   generations_number=generations_number,
                                   n_jobs=parameter_n_jobs)
      start_time=time.time()
      ga1.fit(X, y)
     
      end_time=time.time()
      total_time=end_time-start_time
      
      df_best_score=ga1.best_score_
      df_best_params = pd.DataFrame(ga1.best_params_, index=[0])
     
     
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[1] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:
      rf_params = {}
      clf = SVC(gamma='scale')

      ga1 = EvolutionaryAlgorithmSearchCV(estimator=clf,
                                   params=hyperparameters,
                                   scoring=str(selected_option_metrcix),
                                   cv=int(cv_input),
                                   verbose=1,
                                   population_size=population_size,
                                   gene_mutation_prob=gene_mutation_prob,
                                   gene_crossover_prob=gene_crossover_prob,
                                   tournament_size=tournament_size,
                                   generations_number=generations_number,
                                   n_jobs=parameter_n_jobs)
      start_time=time.time()
      ga1.fit(X, y)
     
      end_time=time.time()
      total_time=end_time-start_time
      df_best_score=ga1.best_score_
      df_best_params = pd.DataFrame(ga1.best_params_, index=[0])
     
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:
      
      clf = KNeighborsClassifier()

      ga1 = EvolutionaryAlgorithmSearchCV(estimator=clf,
                                   params=hyperparameters,
                                   scoring=str(selected_option_metrcix),
                                   cv=int(cv_input),
                                   verbose=1,
                                   population_size=population_size,
                                   gene_mutation_prob=gene_mutation_prob,
                                   gene_crossover_prob=gene_crossover_prob,
                                   tournament_size=tournament_size,
                                   generations_number=generations_number,
                                   n_jobs=parameter_n_jobs)
      start_time=time.time()
      ga1.fit(X, y)
      end_time=time.time()
      total_time=end_time-start_time
      df_best_score=ga1.best_score_
      df_best_params = pd.DataFrame(ga1.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[3] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:
     
     clf = RandomForestRegressor(random_state=parameter_random_state)

     ga1 = EvolutionaryAlgorithmSearchCV(estimator=clf,
                                   params=hyperparameters,
                                   scoring=str(selected_option_metrcix),
                                   cv=int(cv_input),
                                   verbose=1,
                                   population_size=population_size,
                                   gene_mutation_prob=gene_mutation_prob,
                                   gene_crossover_prob=gene_crossover_prob,
                                   tournament_size=tournament_size,
                                   generations_number=generations_number,
                                   n_jobs=parameter_n_jobs)
     start_time=time.time()
     ga1.fit(X, y)
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=ga1.best_score_
     
     df_best_params = pd.DataFrame(ga1.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:    
     
     clf = SVR(gamma='scale')

     ga1 = EvolutionaryAlgorithmSearchCV(estimator=clf,
                                   params=hyperparameters,
                                   scoring=str(selected_option_metrcix),
                                   cv=int(cv_input),
                                   verbose=1,
                                   population_size=population_size,
                                   gene_mutation_prob=gene_mutation_prob,
                                   gene_crossover_prob=gene_crossover_prob,
                                   tournament_size=tournament_size,
                                   generations_number=generations_number,
                                   n_jobs=parameter_n_jobs)
     start_time=time.time()
     ga1.fit(X, y)
     
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=ga1.best_score_
     
     df_best_params = pd.DataFrame(ga1.best_params_, index=[0])
 
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[6]:
     
     clf = KNeighborsRegressor()

     ga1 = EvolutionaryAlgorithmSearchCV(estimator=clf,
                                   params=hyperparameters,
                                   scoring=str(selected_option_metrcix),
                                   cv=int(cv_input),
                                   verbose=1,
                                   population_size=population_size,
                                   gene_mutation_prob=gene_mutation_prob,
                                   gene_crossover_prob=gene_crossover_prob,
                                   tournament_size=tournament_size,
                                   generations_number=generations_number,
                                   n_jobs=parameter_n_jobs)
     start_time=time.time()
     ga1.fit(X, y)
     
     end_time=time.time()
     total_time=end_time-start_time
     df_best_score=ga1.best_score_

     df_best_params = pd.DataFrame(ga1.best_params_, index=[0])
 
 
 
 
 
 
 

    
     
     

 
 


if btn_hyperp:
   st.subheader("The Hyperparametres is :")
   st.table(df_best_params) 
   st.subheader("best score is :")
   st.write(abs(df_best_score)) 
   st.write(f"\n Time consuming during training process :   {total_time:.2f}   seconds")
   st.write(total_time)
   
 
 

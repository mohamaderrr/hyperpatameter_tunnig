import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,SVR 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

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
#select metric:
options_metric = ['RMSE ', 'MSE', 'ACCURACY','R2']
selected_option = st.selectbox('select metric', options_metric)
#cv:
integer_value = st.number_input('Enter nomber of cros folder', step=1, value=1, format='%d')

import streamlit as st

# hyperparametres for tunnuing 



index = st.text_input("Enter index:")
name = st.text_input("Enter name:")
table_data = []


button_clicked = st.button("add one ")
if button_clicked:
   table_data.append((index, name))
   index =''
   name =''
   

# Display the table
st.table(table_data)
    
import streamlit as st

# Get float input from the user
float_input = st.text_input("Enter ourcentage of trainng data")

# pourcentage of trainng data
try:
    float_value = float(float_input)
    if 0 <= float_value <= 1:
        st.write("Float value:", float_value)
    else:
        st.write("Invalid input. Please enter a float value between 0 and 1.")
except ValueError:
    st.write("Invalid input. Please enter a valid float value.")


#select hyperparametres thechning :
option_hyperp_thechnique = ['grid search  ', 'random search ', 'hyperband serachcv','bayesion Optimization with gaussian process','bayesion Optimization with bo-tpe','gradient descent ','pso','genetic  algorithms']
selected_option_h_t = st.selectbox('select hyperparametres thechnique ', option_hyperp_thechnique)
#progress bar :
import time
st.text("progression :")
# Create a progress bar
progress_bar = st.progress(0)

# Update the progress bar
for i in range(101):
    progress_bar.progress(i)
    time.sleep(0.1)



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
    
class random_serach:
    def __init__(self,model,param_distributions,cv,score):
        self.model=model
        self.param=param_distributions
        self.cv=cv
        self.score=score
        
    def fit(self,x,y):
        self.ran_search=RandomizedSearchCV(self.model,self.param,cv=self.cv,n_iter=100,verbose=2,scoring=self.score,random_state=100,n_jobs=-1)  
        self.ran_search.fit(x,y)
        self.best_parametres=self.ran_search.best_params_
    def hyper_parametres(self):
        return self.best_parametres    
        


#dictionary of user input 

btn_hyperp=st.button('get the hyperpametes')       
from sklearn import datasets
if btn_hyperp:
 inputchoise['machine_learning_algorithm']=selected_option_m_l_a
 inputchoise['hyperparamtres_thechnique']=selected_option_h_t
 inputchoise['metric']=options_metric
 inputchoise['cv']=integer_value  
 rf_params = {
    'n_estimators': [10, 20, 30],'max_depth': [15,20,30,40],"criterion":['gini','entropy']}
    
 print(inputchoise['machine_learning_algorithm'])
 print(inputchoise['hyperparamtres_thechnique'])   
 data = datasets.load_digits()
 X = data.data
 y = data.target
 if inputchoise['machine_learning_algorithm']== options_machine_learning[0] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf = ml_model.RFc()
     grid = GridSearchCV(clf, rf_params, cv=3, scoring='accuracy')
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
     grid = GridSearchCV(clf, rf_params, cv=3, scoring='neg_mean_squared_error')
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])
 if inputchoise['machine_learning_algorithm']== options_machine_learning[2] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf=ml_model.knn()
     rf_params = {
    'n_neighbors': [2, 4, 5, 7, 10]}
     
     grid = GridSearchCV(clf, rf_params, cv=3, scoring='neg_mean_squared_error')
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])   
 if inputchoise['machine_learning_algorithm']== options_machine_learning[4] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     clf=ml_model.RFG()
     rf_params = {
    'n_estimators': [10, 20, 12],'max_depth': [15,20,30,50]}
     grid = GridSearchCV(clf, rf_params, cv=3, scoring='neg_mean_squared_error')
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])    
 if inputchoise['machine_learning_algorithm']== options_machine_learning[5] and inputchoise['hyperparamtres_thechnique']== option_hyperp_thechnique[0]:
     rf_params = {'C': [1,10, 100],"kernel":['poly','rbf','sigmoid'],"epsilon":[0.01,0.1,1]}
     clf = SVR(gamma='scale')
     grid = GridSearchCV(clf, rf_params, cv=3, scoring='neg_mean_squared_error')
     grid.fit(X, y)
     print(grid.best_params_)
     print("MSE:"+ str(-grid.best_score_))
     df_best_params = pd.DataFrame(grid.best_params_, index=[0])           
 else:
     print("js")     
 st.title("the hyperparametes is :")
 st.table(df_best_params) 
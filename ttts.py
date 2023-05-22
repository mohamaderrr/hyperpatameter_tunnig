import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
def random_forest_classifier_params():
    st.subheader("Random Forest Classifier Hyperparameters")
    n_estimators = st.slider("Number of Estimators", min_value=1, max_value=100, value=10)
    criterion = st.selectbox("Criterion", ["gini", "entropy"])
    max_depth = st.slider("Maximum Depth", min_value=1, max_value=10, value=5)
    # Add more hyperparameters as needed

    return {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth
        # Add more hyperparameters as needed
    }

def support_vector_machine_params():
    st.subheader("Support Vector Machine Hyperparameters")
    C = st.slider("Regularization Parameter (C)", min_value=0.1, max_value=10.0, value=1.0)
    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
    gamma = st.selectbox("Kernel Coefficient (gamma)", ["scale", "auto"])
    # Add more hyperparameters as needed

    return {
        "C": C,
        "kernel": kernel,
        "gamma": gamma
        # Add more hyperparameters as needed
    }

# Repeat the above pattern for the other algorithms (K-nearest Neighbors, ANN, etc.)
algorithm = st.selectbox("Select an Algorithm", ["Random Forest Classifier", "Support Vector Machine", "K-nearest Neighbors", "ANN"])
if algorithm == "Random Forest Classifier":
    params = random_forest_classifier_params()
elif algorithm == "Support Vector Machine":
    params = support_vector_machine_params()
# Add elif statements for the other algorithms

# Display the selected hyperparameters
st.write("Selected Hyperparameters:", params)

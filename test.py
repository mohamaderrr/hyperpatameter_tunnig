import streamlit as st
import pandas as pd
# Create six columns using the `beta_columns` function
col1, col2, col3, col4, col5, col6 = st.columns(6)

# Column 1
with col1:
 uploaded_file = st.file_uploader("Upload CSV", type="csv")
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
import streamlit as st

# Add custom CSS to set the background color
import streamlit as st

import streamlit as st
import pandas as pd

# Create a DataFrame for the table
data = pd.DataFrame({
    'Name': ['John', 'Jane', 'Tom', 'Emily'],
    'Age': [25, 30, 27, 32]
})

# Display the table
table = st.table(data)

# Allow users to edit the table
for i in range(len(data)):
    name = st.text_input(f"Name {i+1}", value=data.at[i, 'Name'])
    age = st.number_input(f"Age {i+1}", min_value=0, value=data.at[i, 'Age'])
    data.at[i, 'Name'] = name
    data.at[i, 'Age'] = age

# Show the updated table
st.write("Updated Table:", table)

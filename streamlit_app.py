import streamlit as st
import pandas as pd

st.write("""
# Simple Streamlit App
Hello *world* population!
""")

world_population_data_df = pd.read_csv("data/placeholder_data/world_population.csv")

# Select only the population columns
population_columns_indexes = world_population_data_df.columns[
    world_population_data_df.columns.str.match(r'\d{4} Population')  # format: 'XXXX Population'
]

# Create a new DataFrame containing the sums of selected columns and reverse the order of both columns and values
total_population = world_population_data_df[population_columns_indexes].sum()[::-1]

# Convert the Series back to a DataFrame
total_population_df = total_population.to_frame().T

# Add a name to indicate it's the sum of all populations
total_population_df.index = ['Total Population']

st.line_chart(total_population_df)

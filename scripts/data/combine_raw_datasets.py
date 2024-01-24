import pandas as pd
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the paths to the CSV files relative to the script's directory
data_path_1 = os.path.join(script_dir, '..',  '..', 'data', 'raw', 'tweets_dataset_1.csv')
data_path_2 = os.path.join(script_dir, '..',  '..', 'data', 'raw', 'tweets_dataset_2.csv')

# Load the CSV file into DataFrame
df_1 = pd.read_csv(data_path_1)
df_2 = pd.read_csv(data_path_2)

print(df_1)
print(df_2)

# Filter and transform df_1
df_1_filtered = df_1[(df_1['Language'] == 'en') & (df_1['Label'] != 'litigious')]
df_1_filtered['Sentiment'] = df_1_filtered['Label'].replace({'uncertainty': 'neutral'})

# Select relevant columns from df_1
df_1_final = df_1_filtered[['Text', 'Sentiment']]

# Filter and transform df_2
df_2_filtered = df_2
df_2_filtered['Sentiment'] = df_2['sentiment']
df_2_filtered['Text'] = df_2['text']

# Select relevant columns from df_2
df_2_final = df_2[['Text', 'Sentiment']]

# Concatenate both DataFrames
combined_df = pd.concat([df_1_final, df_2_final], ignore_index=True)

# Print the combined DataFrame
print(combined_df)

# Save the combined DataFrame to a CSV file in the 'data/clean' directory
combined_data_path = os.path.join(script_dir, '..', '..', 'data', 'raw', 'combined_tweets_dataset.csv')
combined_df.to_csv(combined_data_path, index=False)

# Print the combined DataFrame save path
print(f"\nCombined DataFrame saved to: {combined_data_path}")

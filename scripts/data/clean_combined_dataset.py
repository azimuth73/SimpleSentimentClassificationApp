import pandas as pd
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the paths
combined_data_path = os.path.join(script_dir, '..', '..', 'data', 'raw', 'combined_tweets_dataset.csv')
clean_data_path = os.path.join(script_dir, '..', '..', 'data', 'clean', 'combined_cleaned_tweets_dataset.csv')

# Load the combined DataFrame
combined_df = pd.read_csv(combined_data_path)


# Perform preprocessing steps here

# Remove missing values and duplicates
combined_df.dropna(subset=['Text'], inplace=True)
combined_df.drop_duplicates(inplace=True)

# Save the cleaned DataFrame to a new CSV file
combined_df.to_csv(clean_data_path, index=False)

# Print a message indicating the completion of the preprocessing
print(f"Preprocessing completed. Cleaned data saved to: {clean_data_path}")

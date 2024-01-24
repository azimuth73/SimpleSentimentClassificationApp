import pandas as pd
import os
from utils.text_preprocessing import preprocess_text

# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the paths
clean_data_path = os.path.join(script_dir, '..', '..', 'data', 'clean', 'combined_cleaned_tweets_dataset.csv')
preprocessed_data_path = os.path.join(script_dir, '..', '..', 'data', 'clean', 'combined_preprocessed_tweets_dataset.csv')

# Load the cleaned DataFrame
clean_df = pd.read_csv(clean_data_path)

# Apply the preprocess_text function to each entry in the 'Text' column
clean_df['Text'] = clean_df['Text'].apply(preprocess_text)

# Save the preprocessed DataFrame to a new CSV file
clean_df.to_csv(preprocessed_data_path, index=False)

# Print a message indicating the completion of the preprocessing
print(f"Text preprocessing completed. Preprocessed data saved to: {preprocessed_data_path}")

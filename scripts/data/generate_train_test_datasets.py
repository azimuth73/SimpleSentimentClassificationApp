import pandas as pd
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load the combined DataFrame
data_path = os.path.join(script_dir, '..', '..', 'data', 'clean', 'combined_preprocessed_tweets_dataset.csv')
df = pd.read_csv(data_path)

# Remove missing values and duplicates
df.dropna(subset=['Text'], inplace=True)
df.drop_duplicates(inplace=True)


# Generating the train and test datasets
# The dataset shouldn't be larger than 100MB
TRAIN_CATEGORY_SIZE = 27_000
TEST_CATEGORY_SIZE = 7_000

# Group by category
grouped_df = df.groupby('Sentiment')

# Initialize empty DataFrames for train and test sets
train_df = pd.DataFrame()
test_df = pd.DataFrame()
# TODO:
#  Need to make a validation dataset
#  A better approach to the current one is to take a more representative sample of data based on average text siz
#  Need to make wordcloud for each of the datasets and the whole one


# Iterate over each category
for category, group in grouped_df:
    # Sort entries based on text size
    sorted_group = group.sort_values(by='Text', key=lambda x: x.str.len(), ascending=False)

    entries = sorted_group.head(TRAIN_CATEGORY_SIZE+TEST_CATEGORY_SIZE)

    # Shuffle entries
    shuffled_entries = entries.sample(frac=1, random_state=42)

    # Split into train and test entries
    train_entries = shuffled_entries.head(TRAIN_CATEGORY_SIZE)
    test_entries = shuffled_entries.tail(TEST_CATEGORY_SIZE)

    # Concatenate the selected entries to the train and test DataFrames
    train_df = pd.concat([train_df, train_entries], ignore_index=True)
    test_df = pd.concat([test_df, test_entries], ignore_index=True)


# Shuffle the train and test DataFrames
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the train and test datasets
train_data_path = os.path.join(script_dir, '..', '..', 'data', 'train_dataset.csv')
test_data_path = os.path.join(script_dir, '..', '..', 'data', 'test_dataset.csv')

train_df.to_csv(train_data_path, index=False)
test_df.to_csv(test_data_path, index=False)

# Check the size of the CSV file
csv_train_size = os.path.getsize(train_data_path)
csv_test_size = os.path.getsize(test_data_path)

print(train_df.info)
print(test_df.info)

print(f'csv_train_size={csv_train_size/1024**2}')
print(f'csv_test_size={csv_test_size/1024**2}')




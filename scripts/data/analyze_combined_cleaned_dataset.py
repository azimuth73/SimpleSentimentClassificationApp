import pandas as pd
import os
import matplotlib.pyplot as plt

# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load the combined DataFrame
combined_data_path = os.path.join(script_dir, '..', '..', 'data', 'clean', 'combined_cleaned_tweets_dataset.csv')
combined_df = pd.read_csv(combined_data_path)

# Display basic information about the DataFrame
print("Basic Information about the Combined DataFrame:")
print(combined_df.info())

# Check for missing values
missing_values = combined_df.isnull().sum()

# Display missing values
print("\nMissing Values:")
print(missing_values)

# Analyze sentiment distribution
sentiment_counts = combined_df['Sentiment'].value_counts()

# Display sentiment distribution
print("\nSentiment Distribution:")
print(sentiment_counts)

# Plot a bar chart of sentiment distribution
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['red', 'green', 'blue', 'gray'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Check maximum length of sentences and characters
max_sentence_length = combined_df['Text'].apply(lambda x: len(x.split())).max()
max_characters = combined_df['Text'].apply(len).max()

print(f"\nMaximum Sentence Length: {max_sentence_length} words")
print(f"Maximum Characters: {max_characters} characters")

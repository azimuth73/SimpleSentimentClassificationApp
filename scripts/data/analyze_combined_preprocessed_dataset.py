import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load the combined DataFrame
combined_data_path = os.path.join(script_dir, '..', '..', 'data', 'clean', 'combined_preprocessed_tweets_dataset.csv')
combined_df = pd.read_csv(combined_data_path)

# Remove missing values and duplicates
combined_df.dropna(subset=['Text'], inplace=True)
combined_df.drop_duplicates(inplace=True)

# Display basic information about the DataFrame
print(combined_df.info())

# Analyze sentiment distribution
sentiment_counts = combined_df['Sentiment'].value_counts()
print(sentiment_counts)

# Plot a bar chart of sentiment distribution
plt.figure(figsize=(8, 8))
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


# Combine all the text in your dataset
text = ' '.join(combined_df['Text'])

# Create a WordCloud object
wordcloud = WordCloud(width=1024, height=1024, background_color='white').generate(text)

# Display the generated word cloud using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

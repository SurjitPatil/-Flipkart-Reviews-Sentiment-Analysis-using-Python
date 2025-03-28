import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# --- Task 1: Importing the data ---
df = pd.read_csv(r"C:\Users\SURJIT PATIL\Downloads\flipkart_reviews.csv")
df

# --- Task 2: Cleaning Up with dropna() ---
cleaned_df = df.dropna()

# --- Task 3: Transforming Flipkart Reviews into Pie ---
ratings = cleaned_df['Rating'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(ratings, labels=ratings.index, autopct='%1.1f%%', startangle=140, 
        colors=['gold', 'lightblue', 'lightgreen', 'red', 'purple'])
plt.title("Flipkart Product Ratings Distribution")
plt.show()

# --- Task 4: Unmasking Customer Emotions with NLTK ---
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

data = pd.DataFrame(cleaned_df['Review'])

data['Positive'] = data['Review'].apply(lambda x: sia.polarity_scores(str(x))['pos'])
data['Negative'] = data['Review'].apply(lambda x: sia.polarity_scores(str(x))['neg'])
data['Neutral'] = data['Review'].apply(lambda x: sia.polarity_scores(str(x))['neu'])

# --- Task 5: Turning Customer Emotions into Numbers ---
x = round(data['Positive'].sum(), 2)
y = round(data['Negative'].sum(), 2)
z = round(data['Neutral'].sum(), 2)
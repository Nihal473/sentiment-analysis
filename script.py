import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("IMDB dataset.csv")  # Update with your correct file path
df = df[['review', 'sentiment']]  # Assume the columns are 'review' and 'sentiment'
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})  # Encode sentiments

# Step 2: Split the data
X = df['review']
y = df['sentiment']

# Convert text to features using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# Step 3: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Make a prediction on the test set
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Take input from the user for review
user_input = input("Enter a movie review: ")

# Step 7: Predict sentiment based on the user's input
def predict_sentiment(input_text):
    input_vec = vectorizer.transform([input_text])  # Vectorize the input
    prediction = model.predict(input_vec)  # Predict sentiment
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    print(f"The sentiment of your review is: {sentiment}")

# Call the function to predict sentiment
predict_sentiment(user_input)

# Step 8: Simplified Visualization (Accuracy Bar Plot)
def simple_visualization(accuracy):
    # Create a very simple bar plot to show model accuracy
    plt.bar(["Accuracy"], [accuracy], color='blue')
    plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy")
    plt.show()

# Call the function to visualize the model's accuracy
simple_visualization(accuracy)



### Movie Review Sentiment Analysis with Naive Bayes

This is a **Sentiment Analysis** project using the **Naive Bayes** algorithm to classify movie reviews as either **positive** or **negative**. The project uses a dataset of 50,000 IMDB movie reviews, where each review is labeled as positive or negative.

### Project Features:

- **Text Preprocessing**: The raw text reviews are transformed into a format that can be used by machine learning algorithms using **CountVectorizer** (bag-of-words model).
- **Model Training**: The project uses the **Multinomial Naive Bayes** classifier to train the model on the preprocessed text data.
- **Model Evaluation**: The model's performance is evaluated using **accuracy** on a test set of movie reviews.
- **User Input**: The user can enter a custom movie review, and the model will predict whether the sentiment is **positive** or **negative**.
- **Simple Visualization**: After the sentiment prediction, the accuracy of the model is displayed in a **basic bar chart** using **Matplotlib**.

### Tools and Libraries:

- **Python**: The project is implemented in Python, a popular language for machine learning.
- **Scikit-learn**: Used for text vectorization, model training, and evaluation.
- **Matplotlib**: Used for visualizing the model's accuracy.
- **Pandas**: Used for data manipulation and handling the dataset.
- **Numpy**: Used for numerical computations.

### Dataset:

The project uses the **IMDB movie reviews dataset**, which contains 50,000 labeled reviews (positive and negative). You can download the dataset directly from the [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### How to Run:

1. Clone the repository to your local machine:
    
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-naive-bayes.git
    
    ```
    
2. Install the required dependencies:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
3. Download the **IMDB dataset** (`IMDB dataset.csv`) and place it in the project directory.
4. Run the script to train the model and classify movie reviews:
    
    ```bash
    python sentiment_analysis.py
    
    ```
    

### Example Input/Output:

**Input**: `Enter a movie review: This movie was amazing! The plot was great and the acting was top-notch.`**Output**: `The sentiment of your review is: Positive`

After the sentiment prediction, a **bar chart** displaying the model's accuracy will be shown.

bash
Copy
Edit
git clone https://github.com/yourusername/sentiment-analysis-naive-bayes.git
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download the IMDB dataset (IMDB dataset.csv) and place it in the project directory.

Run the script to train the model and classify movie reviews:

bash
Copy
Edit
python sentiment_analysis.py
Example Input/Output:
Input: Enter a movie review: This movie was amazing! The plot was great and the acting was top-notch. Output: The sentiment of your review is: Positive

After the sentiment prediction, a bar chart displaying the model's accuracy will be shown.


# Reddit Text Classification Using Machine Learning

## Introduction

### Overview of the Project
In this project, we aim to classify Reddit posts into different topics (or subreddits) using a machine learning approach. We take a set of posts from various subreddits—r/datascience, r/machinelearning, r/physics, r/astrology, and r/conspiracy—and build a model to automatically identify to which subreddit a given post belongs.

### Why Text Classification?
Text classification is a common natural language processing (NLP) task where unstructured text data is categorized into predefined labels. In a practical scenario, such a system can help:
- Automatically route support tickets (for customer service).
- Filter messages and flag potentially harmful content.
- Organize large quantities of text data for information retrieval.

In our case, classifying Reddit posts can help us understand what topics users are discussing, or build recommendation engines to group content by theme.

## Data Extraction

### Using PRAW to Fetch Reddit Data
To work with Reddit data, we use the PRAW (Python Reddit API Wrapper) library, which provides a clean Pythonic interface to the Reddit API. Below is the snippet where we install and import libraries, then authenticate to Reddit and fetch data:

```python
!pip install asyncpraw
!pip install config

import re
import praw
import config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure and authenticate with Reddit API
api_reddit = praw.Reddit(
    client_id="FmxQ8F3VnxZ9cUUoFuT6tQ",
    client_secret="AzNydZ0MD0TO_JPLQsi8swso5J8slw",
    password="testeapi2023",
    user_agent="TheyloApp",
    username="korn_tac"
)

assuntos = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy']
data = []
labels = []
```

### Key Points:
- **API Credentials:** We specify client_id, client_secret, username, and password which Reddit requires for OAuth2-based authentication.
- **Subreddits:** We define a list of subreddit names to fetch data from.

### Data Structures:
- `data` will store the text content of each post.
- `labels` will store the corresponding label index (i.e., which subreddit the text belongs to).




## Filtering and Masking Posts
We sometimes need posts that have enough textual content. The code below shows how we filter out very short posts using a custom lambda function:

```python
char_count = lambda post: len(re.sub('\W|\d', '', post.selftext))
mask = lambda post: char_count(post) >= 100
```

### What’s happening?
- We remove non-alphanumeric characters (`\W|\d`) from the post text to count only meaningful characters.
- If the post has at least 100 such characters, we consider it “long enough” for our classification task.

## Fetching Data
Next, we loop through each subreddit, fetch new posts, filter out short ones, and store the text along with numeric labels:

```python
for i, assunto in enumerate(assuntos):
    subreddit_data = api_reddit.subreddit(assunto).new(limit=1000)
    posts = [post.selftext for post in filter(mask, subreddit_data)]
    data.extend(posts)
    labels.extend([i] * len(posts))
    print(f"Número of posts about r/{assunto}: {len(posts)}", f"
Sample of one post: {posts[0][:600]}...
", "_" * 80 + '
')
```

### Key Points:
- `subreddit_data.new(limit=1000)` fetches the latest 1000 posts from each subreddit.
- We apply our mask function to include only posts that pass the 100-character threshold.
- We print some samples for verification and debugging.



## Data Preprocessing
Once we have our data from Reddit, we need to split it into training and testing sets, and then prepare the text data for our model.

### Train/Test Split
Splitting the dataset helps us prevent overfitting and gauge how the model will perform on unseen data.

```python
TEST_SIZE = 0.2
RANDOM_STATE = 0

X_treino, X_teste, y_treino, y_teste = train_test_split(
    data, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)
print(f"{len(y_teste)} test samples.")
```

#### What’s happening?
- `test_size=0.2` means 20% of the data is held out for testing.
- `random_state=0` ensures reproducibility.

### Text Cleaning and Vectorization
Before feeding text data into a machine learning model, we need to clean and transform it into numerical features.

```python
pattern = r'\W|\d|http.*\s+|www.*\s+'
preprocessor = lambda text: re.sub(pattern, ' ', text)

vectorizer = TfidfVectorizer(
    preprocessor=preprocessor,
    stop_words='english',
    min_df=1
)

decomposition = TruncatedSVD(n_components=1000, n_iter=30)
pipeline = [('tfidf', vectorizer), ('svd', decomposition)]
```

#### Key Points:
- **Preprocessing:**
  - We remove unwanted patterns such as non-word characters, digits, and URLs (http.* or www.*) using a regular expression.
  - We also use `stop_words='english'` to remove common words (like “the”, “and”, “is”) that do not carry unique meaning.
- **TF-IDF Vectorization:**
  - TF-IDF (Term Frequency-Inverse Document Frequency) transforms text into a feature matrix where each dimension corresponds to the importance of a token in a document relative to other documents.
- **Dimensionality Reduction:**
  - Truncated SVD reduces the feature space to 1,000 components, helping the model cope with the high dimensionality typical in text data and potentially speeding up training.




## Model Building
We will train three different models to see which works best:

```python
modelo_1 = KNeighborsClassifier(n_neighbors=4)
modelo_2 = RandomForestClassifier(random_state=RANDOM_STATE)
modelo_3 = LogisticRegressionCV(cv=3, random_state=RANDOM_STATE)
all_models = [("KNN", modelo_1), ("RandomForest", modelo_2), ("LogReg", modelo_3)]
```

### Why These Models?
- **K-Nearest Neighbors (KNN):** A simple instance-based model that classifies a sample based on the classes of its nearest neighbors in the feature space.
- **Random Forest:** An ensemble of decision trees, typically performing well on a variety of tasks due to its robustness and ability to model complex interactions.
- **Logistic Regression with Cross-Validation (CV):** A strong baseline for text classification, with built-in cross-validation to help fine-tune regularization.

## Model Evaluation
### Training and Generating Predictions
We use a pipeline that chains together our text preprocessing and model training:

```python
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

resultados = []

for name, modelo in all_models:
    pipe = Pipeline(pipeline + [(name, modelo)])
    pipe.fit(X_treino, y_treino)
    y_pred = pipe.predict(X_teste)
    report = classification_report(y_teste, y_pred)
    print(f"{name} report\n", report)
    resultados.append([modelo, {'model': name, 'predictions': y_pred, 'report': report}])
```

### Key Points:
- **Pipeline:** We sequentially apply the TF-IDF transform, dimensionality reduction (SVD), and finally pass the data to our classifier.
- **Classification Report:** Shows important metrics like precision, recall, F1-score, and support for each class.

## Visualizing the Results
To understand our dataset distribution and how well our model performed, we create several visualizations.

```python
_, counts = np.unique(labels, return_counts=True)
sns.set_theme(style="whitegrid")

plt.figure(figsize=(15, 6), dpi=120)
plt.title("Number of post per subjects")
sns.barplot(x=assuntos, y=counts)
plt.legend([' '.join([f.title(), f"- {c} posts"]) for f, c in zip(assuntos, counts)])
plt.show()
```

### What’s happening?
- We count how many posts came from each subreddit.
- We plot a simple bar chart to illustrate the data imbalance (if any).




## Confusion Matrix
A confusion matrix helps us see where our model is making mistakes:

```python
def plot_confusion(result):
    print("Classification report\n", result['report'])
    y_pred = result['predictions']
    conf_matrix = confusion_matrix(y_teste, y_pred)

    _, test_counts = np.unique(y_teste, return_counts=True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100

    plt.figure(figsize=(9, 8), dpi=120)
    plt.title(result['modelo'].upper() + " Resultados")
    plt.xlabel("Valor Real")
    plt.ylabel("Previsão do Modelo")
    ticklabels = [f"r/{sub}" for sub in assuntos]
    sns.heatmap(data=conf_matrix_percent,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
                annot=True,
                fmt='.2f')
    plt.show()

for result in resultados:
    plot_confusion(result[1])
```

### Key Points:
- We compute the confusion matrix for each model’s predictions vs. actual labels.
- We normalize it by the true label counts to express how often each class gets confused with another in percentage terms.
- **Heatmap:** The heatmap helps pinpoint if a model frequently misclassifies one particular subreddit as another.

## Conclusion
### What We Achieved
- **Data Collection & Preparation:** We successfully gathered data from multiple subreddits using PRAW, filtered it, and prepared it by removing unwanted characters.
- **Feature Engineering:** We used a TF-IDF vectorizer and Truncated SVD to handle the high dimensionality of text data.
- **Model Comparison:** We trained and evaluated three popular classifiers—KNN, Random Forest, and Logistic Regression—comparing their performances using classification reports and confusion matrices.

### Potential Improvements
- **Hyperparameter Tuning:** While we did some basic cross-validation with Logistic Regression, further tuning of all models (e.g., grid search or Bayesian optimization) could lead to improved results.
- **Advanced Embeddings:** We could replace TF-IDF with advanced word embeddings (like Word2Vec, GloVe, or transformer-based embeddings) to capture more nuanced language features.
- **Data Augmentation:** Since Reddit post content can be imbalanced or limited in certain topics, techniques like oversampling or leveraging large language models for data augmentation might further improve performance.

By refining these areas, we can build a more robust and accurate text classification pipeline for Reddit data—or any other text-based dataset.

Thank you for reading! If you have any questions or suggestions, feel free to leave a comment or reach out. Happy coding and happy classification!


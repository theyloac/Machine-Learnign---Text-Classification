
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

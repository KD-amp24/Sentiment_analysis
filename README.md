# Sentiment_analysis
Python pipeline that uses NLTK to analyze the sentiment and language patterns of phishing vs. safe emails.

What This Pipeline Does
This pipeline analyzes phishing and safe emails to understand how their language differs, how emotional tone differs and whether phishing emails use more urgency or pressure. It does this in three major stages:

PART 1 - Loading and Inspecting the Data
1.	Loading the Dataset: The script uses pandas to load the CSV file. Once loaded, pandas stores everything in a DataFrame, which works like a spreadsheet inside Python.

2.	Inspecting the Dataset: The script prints the dataset shape and the first 5 rows. This confirms that labels are correct and that emails are loaded properly.

3.	Counting Email Types: It then counts how many phishing vs safe emails exist.

4.	Printing Example Emails: The script prints 3 phishing and 3 safe emails.

PART 2 - Text Preprocessing (Cleaning the Emails)
Before analyzing language, the emails must be cleaned. Raw text contains capital letters, punctuation, common filler words, formatting noise etc. Computers work better with structured tokens.
1.	Tokenization: The email is split into individual words.
Example:
Urgent: The account has been compromised.
Becomes:
[‘Urgent’, ‘:’, ‘The’, ‘account’, ‘has’, ‘been’, ‘compromised’, ‘.’]

2.	Lowercasing: All words are converted to lowercase.

3.	Removing Punctuation: Symbols like ‘:’, ‘.’, ‘,’, ‘!’ etc. are removed because they don’t carry strong meaning for this analysis.

4.	Removing Stopwords: Stopwords are very common words like ‘the’, ‘is’, ‘has’, ‘been’ etc. They appear in almost every sentence and don’t help distinguish phishing from safe emails.
After cleaning:
‘Urgent: The account has been compromised.’
Becomes:
[‘urgent’, ‘account’, ‘compromised’, ‘verify’, ‘information’, ‘immediately’, ‘restore’, ‘access’]

PART 3 - Sentiment Analysis (Using VADER)
This measures the emotional tone.
The script uses:
NLTK’s VADER (Valence Aware Dictionary for Sentiment Reasoning)
VADER assigns four scores to each email:
•	positive
•	negative
•	neutral
•	compound (overall score)
The pipeline uses the compound score, which ranges from -1 to 1.

The script calculates the average sentiment for phishing emails and average sentiment for safe emails.
This is a structured linguistic analysis system that cleans email text, extracts meaningful language features, computes emotional tone, and statistically compares phishing vs safe communication patterns.

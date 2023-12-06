"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Packages needed for cleaning
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
nltk.download('punkt')
nltk.download('wordnet')
import unicodedata
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
import string
import re

# Vectorizer
news_vectorizer = open("./Models/Train/TFIDF_Vec_Train.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your df_clean data
df_clean = pd.read_csv("./train.csv")

# Cleaning data

# Create functions to extract twitter handles, hastags, and retweet handles
'''Determine if there is a retweet within the tweet'''
def is_retweet(tweet):
    word_list = tweet.split()
    if "RT" in word_list:
        return 1
    else:
        return 0

''' Function to extract retweet handles from tweet '''
def get_retweet(tweet):
    word_list = tweet.split()
    if word_list[0] == "RT":
        handle = word_list[1]
    else:
        handle = ''
        
    handle = handle.replace(':', "")

    return handle


'''Count the number of hashtags within the tweet'''
def count_hashtag(tweet):
    count = 0
    word_list = tweet.split()
    for word in word_list:
        if word[0] == '#':
            count +=1
    
    return count

'''Extract the hashtags within the tweet'''
def get_hashtag(tweet):
    hashtags = []
    word_list = tweet.split()
    for word in word_list:
        if word[0] == '#':
            hashtags.append(word)
    
    returnstr = ""
    for tag in hashtags:
        returnstr += " " + tag

    return returnstr


'''Count the number of mentions within the tweet'''
def count_mentions(tweet):
    count = 0
    word_list = tweet.split()
    if "RT" in word_list:
        count += -1 # Remove mention contained in retweet form consideration
        
    for word in word_list:
            if word[0] == '@':
                count +=1
    if count == -1:
        count = 0
    return count

'''Extract the mentions within the tweet'''
def get_mentions(tweet):
    mentions = []
    word_list = tweet.split()
    if "RT" in word_list:
        word_list.pop(1) # Retweets don't count as mentions, so we remove the retweet handle from consideration

    for word in word_list:
        if word[0] == '@':
            mentions.append(word)
    
    returnstr = ""
    for handle in mentions:
        returnstr += " " + handle

    return returnstr

# function to count the number of web links within tweet
def count_links(tweet):
		count = tweet.count("https:")
		return count 

# Count number of exclamation marks within tweet:
def exclamation_count(tweet):
    count = tweet.count('!')
    return count

# Count number of question marks within tweet:
def question_count(tweet):
    count = tweet.count('?')
    return count

# Convert all text in the 'message' column to lowercase
df_clean['message'] = df_clean['message'].str.lower()

# Apply the functions created above

# Get retweet status and handle
df_clean["is_retweet"] = df_clean["message"].apply(is_retweet)
df_clean["retweet_handle"] =  df_clean["message"].apply(get_retweet)

# Get hashtag count and extract hashtags
df_clean["hashtag_count"] = df_clean["message"].apply(count_hashtag)
df_clean["hashtags"] =  df_clean["message"].apply(get_hashtag)

# Get mention count and extract mentions
df_clean["mention_count"] = df_clean["message"].apply(count_mentions)
df_clean["mentions"] =  df_clean["message"].apply(get_mentions)

# Get number of links
df_clean["link_count"] = df_clean["message"].apply(count_links)

# Get number of question marks
df_clean["question_count"] =  df_clean["message"].apply(question_count)

# Get number of exclamation marks
df_clean["exclamation_count"] =  df_clean["message"].apply(exclamation_count)

# Define the URL pattern using a regular expression
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

# Substitute URLs with a generic term
subs_url = 'url-web'

# Replace URLs in the 'message' column of df_clean
df_clean['message'] = df_clean['message'].replace(to_replace=pattern_url, value=subs_url, regex=True)

# Functions to remove hamdles, hashtags, and retweets

# Remove handles from tweet:
def remove_handles(tweet):
    wordlist = tweet.split()
    for word in wordlist:
        if word[0] == '@':
            wordlist.remove(word)
    returnstr = ''
    for word in wordlist:
        returnstr += word + " "
    return returnstr

# Remove handles from tweet:
def remove_hashtags(tweet):
    wordlist = tweet.split()
    for word in wordlist:
        if word[0] == '#':
            wordlist.remove(word)
    returnstr = ''
    for word in wordlist:
        returnstr += word + " "
    return returnstr

# Remove RT from tweet:
def remove_rt(tweet):
    wordlist = tweet.split()
    for word in wordlist:
        if word == 'rt' or word=='RT':
            wordlist.remove(word)
    returnstr = ''
    for word in wordlist:
        returnstr += word + " "
    return returnstr

# Remove handles from tweet
df_clean["message"] = df_clean['message'].apply(remove_handles)
# Remove hashtags from tweet
df_clean["message"] = df_clean['message'].apply(remove_hashtags)
# Remove RT from tweet
df_clean["message"] = df_clean['message'].apply(remove_rt)

# Function to remove numbers from tweet
def remove_numbers(tweet):
    new_string = re.sub(r'[0-9]', '', tweet)
    return new_string

# Apply the function to the data
df_clean["message"] =df_clean['message'].apply(remove_numbers)

# Write function to replace contractions:
def fix_contractions(tweet):
    expanded_words = []
    for word in tweet.split():
        expanded_words.append(contractions.fix(word))
    
    returnstr = " ".join(expanded_words)
    return returnstr

# Apply function to tweet message:
df_clean["message"] = df_clean['message'].apply(fix_contractions)

# Function to remove punctuations
def remove_punctuation(tweet):
    return ''.join([l for l in tweet if l not in string.punctuation and l != '’'])

# Apply the remove_punctuation function to the 'message' column in df_clean
df_clean['message'] = df_clean['message'].apply(remove_punctuation)

# Create function to replace strange characters in data with closest ascii equivalent
def clean_tweet(tweet):
    # Normalize the tweet to remove diacritics and other special characters
    normalized_tweet = unicodedata.normalize('NFKD', tweet)
    
    # Remove or replace any remaining unwanted characters
    cleaned_tweet = normalized_tweet.encode('ascii', 'ignore').decode('utf-8')
    
    return cleaned_tweet.lower()

df_clean["message"] =df_clean['message'].apply(clean_tweet)

# Remove common English stop words from the data
def remove_stop_words(tweet):
    words = tweet.split()     
    return ' '.join([t for t in words if t not in stopwords.words('english')])

# Remove stop words from data
df_clean["message"] = df_clean['message'].apply(remove_stop_words)

# Make dataframe of all word counts in the data
df_wordcounts = pd.DataFrame(df_clean['message'].str.split(expand=True).stack().value_counts())
df_wordcounts.reset_index(inplace=True)
df_wordcounts.rename(columns={"index": "word", 0:"count"}, inplace=True)

# Extract unique words from data
df_unique_words = df_wordcounts[df_wordcounts["count"]==1]
df_unique_words

# Investigate amount of unique words
print(len(df_unique_words), "out of", len(df_wordcounts), "words in our dataset appears only once, i.e.", str(round(len(df_unique_words)/len(df_wordcounts)*100, 2)) +"%", "of words used are unique")

# Make list of unique words
unique_wordlist = list(df_unique_words["word"])

# Function to remove unique words from data
def remove_unique_words(tweet):
    words = tweet.split()     
    return ' '.join([t for t in words if t not in unique_wordlist])

# Applying the function
df_clean["message"] =df_clean['message'].apply(remove_unique_words)

# Tolkenization:

tokeniser = TreebankWordTokenizer()

df_clean['message'] = df_clean['message'].apply(tokeniser.tokenize)

# Lemmatization:

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# Create function to lemmatize tweet content
def tweet_lemma(tweet, lemmatizer):
    list_of_lemmas = [lemmatizer.lemmatize(word) for word in tweet] 
    return " ".join(list_of_lemmas) 

df_clean["message"] = df_clean["message"].apply(tweet_lemma, args=(lemmatizer, ))

# Put hashtags, handles, and retweets back into 'messages'

# Function to add retweets to message
def add_rt_handle(row):
    if row["retweet_handle"] == "":
        ret = row["message"]
    else:
        ret =  row["message"] + " rt_" + row["retweet_handle"]
    return ret
# Applying the function
df_clean["message"] = df_clean.apply(add_rt_handle, axis=1)

# Function to add retweets to message
def add_hashtag(row):
    if row["hashtags"] == "":
        ret = row["message"]
    else:
        ret =  row["message"] + " " + row["hashtags"]
    return ret
# Applying the function
df_clean["message"] = df_clean.apply(add_hashtag, axis=1)

# Function to add mentions to message
def add_rt_handle(row):
    if row["mentions"] == "":
        ret = row["message"]
    else:
        ret =  row["message"] + " " + row["mentions"]
    return ret

# Applying the function
df_clean["message"] = df_clean.apply(add_rt_handle, axis=1)

# Drop the specified columns
df_clean = df_clean.drop(["retweet_handle", "hashtags", "mentions"], axis=1)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("df_clean Twitter data and label")
		if st.checkbox('Show df_clean data'): # data is hidden if box is unchecked
			st.write(df_clean[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("./Models/LogisticRegression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

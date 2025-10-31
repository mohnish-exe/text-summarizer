import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

def text_summarizer(text, num_sentences=3):
    # Text into sentences
    sentences = sent_tokenize(text)

    # Text into words
    words = word_tokenize(text.lower())

    # Removing stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.casefold() not in stop_words]

    # Calculate word frequencies
    fdist = FreqDist(filtered_words)

    # Assign scores to sentences based on word frequencies
    sentence_scores = [sum(fdist[word] for word in word_tokenize(sentence.lower()) if word in fdist)
                       for sentence in sentences]

    # Create a list of tuples containing sentence index and score
    sentence_scores = list(enumerate(sentence_scores))

    # Sort sentences by scores in descending order
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    # Randomly select the top `num_sentences` sentences for the summary
    random_sentences = random.sample(sorted_sentences, num_sentences)

    # Sort the randomly selected sentences based on their original order in the text
    summary_sentences = sorted(random_sentences, key=lambda x: x[0])

    # Create the summary
    summary = ' '.join([sentences[i] for i, _ in summary_sentences])

    return summary

# Example usage
text = """
Instead of importing the whole nltk module, I simply imported the essential submodules (stopwords, word_tokenize, sent_tokenize, and FreqDist), which reduced unnecessary memory usage and improved efficiency.

To make the code cleaner and easier to read, I deleted self-explanatory comments.

NLTK includes a FreqDist class for calculating word frequencies. I used it directly on the list of words, eliminating the need to manually cycle through each word and update the frequency distribution.

To simplify the code, I used list comprehension instead of a standard for loop to calculate sentence scores and added randomness into the selection process of the top sentences.
"""

summary = text_summarizer(text)
print(summary)

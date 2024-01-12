from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

tweet = "@akhimed it's cold today :( https://www.giantbomb.com/takumi-fujiwara/3005-3803/friends/"

#preprcess tweet
tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    elif word.startswith('http'):
        word = " "
    tweet_words.append(word)

tweet_process = " ".join(tweet_words)

print(tweet_process)

# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)

tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
encoded_tweet = tokenizer(tweet_process, return_tensors='pt')
#output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)
print(scores)

for i in range(len(scores)):
    l = labels[i]
    s = scores[i]
    print(l, s)

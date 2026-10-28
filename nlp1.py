import nltk
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import inaugural
import matplotlib.pyplot as plt
from nltk.probability import ConditionalFreqDist

obama_speech = inaugural.words(fileids='2009-Obama.txt')
text = ' '.join(obama_speech).lower()

cfd = ConditionalFreqDist((word, word) for word in obama_speech)
print(cfd[6])

stopwords = set(STOPWORDS)
stopwords.update(['mr', 'president', 'america', 'americans', 'year', 'must', 'one', 'us', 'nation', 'people'])

wordcloud = WordCloud(
    width=1200,
    height=800,
    background_color='white',
    stopwords=stopwords,
    collocations=False
).generate(text)

fd=nltk.FreqDist(obama_speech)
print(fd.most_common(20))

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('obama_speech_wordcloud.png', dpi=150)
plt.show()
print('Saved word cloud to obama_speech_wordcloud.png')
obamaSpeech = inaugural.words(fileids='2009-Obama.txt')[:20]
print(obamaSpeech)

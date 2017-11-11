import nltk

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
sentence = "At eight o'clock on Thursday morning"
tokens = nltk.word_tokenize(sentence)
# a = nltk.pos_tag_sents(sentence)
print tokens
# print a

text = nltk.word_tokenize("play Disney Sing It! - High School Musical 3: Senior Year")
print text
a = nltk.pos_tag(text)
print a
for tup in a:
    t = tup[0] + "_" + tup[1]
    print t
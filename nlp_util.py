import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# Name-entity recognition using NLTK

# example strings
input_str1: str = "stores near me"
input_str2 = "library near me"
input_str3 = "nearest gas station"
input_str4 = "Nearest Airport"
input_str5 = "african restaurant near me"
input_str6 = "mosque near me chicago"
input_str7 = "mexican restaurants near my location"
input_str8 = "best restaurant near me"
input_str9 = "lodging near arlington heights, IL"

# tokenize
print(ne_chunk(pos_tag(word_tokenize(input_str1))))
print(ne_chunk(pos_tag(word_tokenize(input_str2))))
print(ne_chunk(pos_tag(word_tokenize(input_str3))))
print(ne_chunk(pos_tag(word_tokenize(input_str4))))
print(ne_chunk(pos_tag(word_tokenize(input_str5))))
print(ne_chunk(pos_tag(word_tokenize(input_str6))))
print(ne_chunk(pos_tag(word_tokenize(input_str7))))

query_tokens = word_tokenize(input_str9)  # ['stores', 'near', 'me']

pos_tagged = pos_tag(query_tokens)  # [('stores', 'NNS'), ('near', 'IN'), ('me', 'PRP')]

print(ne_chunk(pos_tagged))

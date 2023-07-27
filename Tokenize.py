# import selfies
from SmilesPE.pretokenizer import atomwise_tokenizer#, kmer_tokenizer
import atomInSmiles
# import spacy
# import re


class smilespetokenize(object):
    def tokenizer(self, sentence):
        return [tok for tok in atomwise_tokenizer(sentence) if tok != " "]

class atomtokenize(object):
    def tokenizer(self, sentence):
        return atomInSmiles.encode(sentence).split()
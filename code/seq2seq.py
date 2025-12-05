from pathlib import Path
import unicodedata
import re
import random

DATA_DIR = Path("../data/")
RAW_DATA_FILE = DATA_DIR / "raw_data" / "en_fr_raw.tsv"
INTERMEDIATE_DATA_FILE = DATA_DIR / "processed_data" / "en_fr.tsv"

SOS_TOKEN = 0
EOS_TOKEN = 1

class Lang:
    def __init__(self, name) -> None:
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.n_words = 2 # counting EOS and SOS as words
    def addWord(self, word):
        if word not in self.word2index:
            self.index2word[self.n_words] = word
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.n_words +=1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

def preprocess_data(input_path = RAW_DATA_FILE, output_path = INTERMEDIATE_DATA_FILE):
    """Removes the extra sentence indices from the Tatoeba data"""
    import csv

    with open(input_path, "r", newline='', encoding='utf-8') as infile, \
        open(output_path, "w", newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        for row in reader:
            writer.writerow([row[1], row[3]])
    return



def unicodeToAscii(s):
    ascii_string = "".join(
        c for c in unicodedata.normalize("NFD", s) # Normal Form D (canonical decomposition)
        if unicodedata.category(c) != "Mn" # excludes Nonspacing_Mark
    )
    return ascii_string

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2):
    print("Reading lines...")

    lines = open(INTERMEDIATE_DATA_FILE, encoding="utf-8")\
            .read().strip().split('\n')

    pairs = [ [normalizeString(s) for s in l.split('\t')] for l in lines]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs



preprocess_data()

def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra')
print(random.choice(pairs))

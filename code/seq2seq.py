import csv
import numpy as np
import torch
from pathlib import Path
import unicodedata
import re
import random
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch import optim, rand

DATA_DIR = Path("../data/")
RAW_DATA_FILE = DATA_DIR / "raw_data" / "en_fr_raw.tsv"
INTERMEDIATE_DATA_FILE = DATA_DIR / "processed_data" / "en_fr.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_STEPS = 10
RANDOM_TRAINING_FRACTION = 75

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
        if unicodedata.category(c) != "Mn" # excludes Non spacing_Mark
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
def filterPair(p):
    return len(p[0].split(' ')) < MAX_STEPS and \
        len(p[1].split(' ')) < MAX_STEPS 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    pairs = filterPairs(pairs) # only consider a subset for faster training
    pairs = random.sample(pairs, len(pairs)//RANDOM_TRAINING_FRACTION)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p = 0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=DEVICE).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_STEPS):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = torch.nn.functional.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'fra')

    n = len(pairs)
    input_ids = np.zeros((n, MAX_STEPS), dtype=np.int32)
    target_ids = np.zeros((n, MAX_STEPS), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        
        inp_ids = inp_ids[:MAX_STEPS-1]
        tgt_ids = tgt_ids[:MAX_STEPS-1]

        inp_ids.append(EOS_TOKEN)
        tgt_ids.append(EOS_TOKEN)

        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(DEVICE),
                               torch.LongTensor(target_ids).to(DEVICE))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader, pairs


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smooth = SmoothingFunction().method1

def compute_bleu(encoder, decoder, pairs, input_lang, output_lang):
    scores = []
    for src, tgt in pairs:
        pred_words, _ = evaluate(encoder, decoder, src, input_lang, output_lang)
        
        # remove <EOS>
        pred = [w for w in pred_words if w != '<EOS>']
        ref = tgt.split()

        bleu = sentence_bleu([ref], pred, smoothing_function=smooth)
        scores.append(bleu)

    return sum(scores) / len(scores)


hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size)

encoder = Encoder(input_lang.n_words, hidden_size).to(DEVICE)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(DEVICE)

train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

encoder.eval()
decoder.eval()
bleu_score = compute_bleu(encoder, decoder, pairs, input_lang, output_lang)
print("BLEU:", bleu_score)
evaluateRandomly(encoder, decoder)

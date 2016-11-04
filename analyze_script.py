from __future__ import unicode_literals, division

import nltk.data
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet, sentiwordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))
POS_TAG_MAP = {'NOUN': wordnet.NOUN,
               'ADJ': wordnet.ADJ,
               'VERB': wordnet.VERB,
               'ADV': wordnet.ADV
               }

# a little memoization for synset word scores
WORD_SCORES = {}

# for replacing contractions post-tokenization
CONTRACTION_MAP = {"'s": "is",
                   "'ll": 'will',
                   "n't": "not",
                   "'ve": "have",
                   "'re": "are",
                   "'m": "am",
                   "ca": "can",
                   "'d": "would"}

lemmatizer = WordNetLemmatizer()
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sid = SentimentIntensityAnalyzer()


def normalize_arr(arr, mn=None, mx=None):
    if not mn:
        mn, mx = min(arr), max(arr)
    return map(lambda x: (x - mn) / (mx - mn), arr)


def replace_contractions(token):
    if token in CONTRACTION_MAP:
        return CONTRACTION_MAP[token]
    return token


def word_senti_score(word, POS):
    """returns nltk sentiwordnet...

    Args:
        word (str): Description
        POS (str): part of speech should be
                   gotta be in NLTK wordnet

    Returns:
        TYPE: pos & neg values... skips neu
    """
    p, n = 0., 0.
    try:
        p, n = WORD_SCORES[(word, POS)]
    except KeyError:
        scores = sentiwordnet.senti_synsets(word, POS)
        if scores:
            p, n = sum([s.pos_score() for s in scores]) / \
                   len(scores), sum([s.neg_score() for s in scores]) / len(scores)
        WORD_SCORES[(word, POS)] = (p, n)
    return p, n



def get_words(sent, word_clean, stop, exceptions=[]):
    """tag and tokenize sentance, do cleanup on words
        and return list of word, POS pairs not in stopwords whithin valid
        parts of speech (default is limited set of POS's that wordnet
        handles).

    Args:
        sent (TYPE): sentence, not tokenized
        word_clean (function): cleaning functions to be run on
                               words after tagging
        stop (List): list of stopwords default is from nltk
        exceptions (list, optional): these words will escape the
                                     lemmatizer.

    Returns:
        List: word, POS pairs
    """
    if exceptions is None:
        exceptions = []
    tagged = pos_tag(word_tokenize(sent), tagset='universal')
    words = map(lambda (x, y): (word_clean(x), y), tagged)
    wn_tagged = [(p[0], POS_TAG_MAP[p[1]])
                 for p in words if p[1] in POS_TAG_MAP]
    res = []
    for (w, t) in wn_tagged:
        if w in exceptions:
            res.append((w, t))
        else:
            lem = lemmatizer.lemmatize(w, t)
            if lem not in stop:
                res.append((lem, t))
    return res


def get_sentences(transcript, labels, regex):
    sentences = dict(((label, []) for label in labels))
    for line in regex.findall(transcript):
        speaker, text = line[0].lower(), ' '.join(line[1:])
        if speaker in labels:
            sentences[speaker] += sentence_tokenizer.tokenize(text)
    return sentences


def get_vocab(sentences, word_getter):
    words = []
    for sentence in sentences:
        score = sid.polarity_scores(sentence)
        s_pos, s_neg = score['pos'], score['neg']
        wrds = word_getter(sentence)
        for (w, POS) in wrds:
            p, n = word_senti_score(w, POS)
            w_pos = 1. * (p + s_pos)
            w_neg = 1. * (n + s_neg)
            words.append((w, POS, w_pos, w_neg))
    unique_words = set([e[0] for e in words])
    vocab = [list(unique_words), [], [], []]
    for u_word in unique_words:
        w_dat = filter(lambda x: x[0] == u_word, words)
        count = len(w_dat)
        vocab[1].append(count)
        p, n = sum([e[-2] for e in w_dat]) / float(count), sum([e[-1]
                                                                for e in w_dat]) / float(count)
        vocab[2].append(p)
        vocab[3].append(n)

    vocab[2] = normalize_arr(vocab[2])
    vocab[3] = normalize_arr(vocab[3])
    return vocab


def get_data(txt, labels, regex, additional_stopwords=[], exceptions=[]):
    txt = txt.replace('\u2019', "'")
    sentences = get_sentences(txt, labels, regex)
    labeled_data = []

    for lab in labels:
        vocab_data = get_vocab(sentences[lab],
                               word_getter=lambda sentence: get_words(sentence,
                                                                      word_clean=lambda x: replace_contractions(
                                                                          x.lower()),
                                                                      stop=additional_stopwords | STOP_WORDS,
                                                                      exceptions=exceptions))
        labeled_data.append((lab, vocab_data))

    data = {}
    for label, (vocab, counts, pos_vals, neg_vals) in labeled_data:
        data[label] = dict(((w, dict(count=counts[i], pos=pos_vals[
            i], neg=neg_vals[i])) for i, w in enumerate(vocab)))

    return data

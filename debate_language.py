from __future__ import unicode_literals
import os
import codecs
import re
import matplotlib.pyplot as plt
import analyze_script
from wordcloud import WordCloud

dat_dir = os.path.join(os.path.dirname(__file__), 'data')
filenames = [f for f in os.listdir(dat_dir) if f.endswith('txt')]


def gen_clouds(data):
    wordclouds = []
    for speaker, voc in data.items():
        counts = [(w, voc[w]['count']) for w in voc]

        def sent_color_function(word=None, font_size=None, position=None,
                                orientation=None, font_path=None, random_state=None):
            """sentiment color generator for WordCloud, red is negative and 
                blue is positive.
            """
            r, g, b = 122,  122, 122
            if voc[word]:
                r +=int(255 * voc[word]['neg'])
                b += int(255 *  voc[word]['pos'])
            return "rgb({}, {}, {})".format(r, g, b)

        wordcloud = WordCloud(  max_font_size = 100,
                                # relative_scaling=.6,
                                width= 800, 
                                height = 400,
                                color_func=sent_color_function).generate_from_frequencies(counts)
        wordclouds.append((speaker, wordcloud))
        wordcloud.to_file(os.path.join(dat_dir, '{}.jpg'.format(speaker)))
    return wordclouds


def show_clouds(clds, n=221):
    for l, cl in clds:
        plt.subplot(n)
        plt.title(l)
        plt.imshow(cl)
        plt.axis("off")
        n += 1
    plt.show()


regex = re.compile(
    r'(?P<speaker>^[A-Za-z]+)\:\s(?P<text>.*)(?:\n\[.*?\]\n)?(?P<contd>.*)', re.MULTILINE)

canidates = ['clinton', 'trump']

# these are a set of words that are added to nltk's stopwords

junk = set([ 'say', 'get', 'think', 'go',  'well', 'come', 'would', 'could', 'look',
             'would', 'want', 'become', 'donald', 'hillary', 'make', 'chris', 'know', 
             'take', 'lot', 'tell', 'way', 'need', 'give', 'see', 'year', 'many', 'talk', 'clinton', 
             'trump', 'really', 'look', 'let', 'much', 'look', 'country', 'president', 'also', 'lester', 
             'people', ])

# these words will escape the lemmatizer
exceptions = ['isis', 'isil', 'sanders']

f_data = ''
clouds = []
for fn in filenames:
    with codecs.open(os.path.join(dat_dir, fn), 'rb', 'UTF-8') as f:
        f_data += f.read().replace('\u2019',
                                   "'").replace(' [Interruption]', '') + '\n'

vocabs = analyze_script.get_data(
    f_data, canidates, regex=regex, additional_stopwords=junk, exceptions=exceptions)



clouds += gen_clouds(vocabs)

shared_words = set(set(tsorted) & set(csorted))

show_clouds(clouds, 221)

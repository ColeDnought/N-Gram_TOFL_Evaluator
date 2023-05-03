import sys
from collections import defaultdict
import math
import os
import os.path

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    result = []
    sequence = list(sequence)
    sequence.insert(0, "START")
    sequence.append("STOP")

    for i in range(2, n):
        sequence.insert(0, "START")

    for j in range(0, len(sequence) - (n - 1)):
        temp = []
        for k in range(0, n):
            temp.append(sequence[j + k])
        gram = tuple(temp)
        result.append(gram)
    return result


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        #sentence = self.generate_sentence(20)



    def count_ngrams(self, corpus):
        self.unigramcounts = {}
        self.bigramcounts = {} 
        self.trigramcounts = {}
        self.total_sentences = 0
        self.total_words = 0

        for sequence in corpus:
            self.total_sentences += 1
            unigrams = get_ngrams(sequence, 1)
            for unigram in unigrams:
                if unigram in self.unigramcounts:
                    self.unigramcounts[unigram] += 1
                else:
                    self.unigramcounts[unigram] = 1
                self.total_words += 1
            bigrams = get_ngrams(sequence, 2)
            for bigram in bigrams:
                if bigram in self.bigramcounts:
                    self.bigramcounts[bigram] += 1
                else:
                    self.bigramcounts[bigram] = 1
            trigrams = get_ngrams(sequence, 3)
            for trigram in trigrams:
                if trigram in self.trigramcounts:
                    self.trigramcounts[trigram] += 1
                else:
                    self.trigramcounts[trigram] = 1
        return

    def raw_trigram_probability(self, trigram):
        if trigram[:2] == ('START', 'START'):
            return self.raw_bigram_probability(trigram[1:])

        if trigram not in self.trigramcounts:
            return 1/len(self.lexicon)

        return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        if bigram not in self.bigramcounts:
            return 1/len(self.lexicon)

        if bigram[0] == 'START':
            return self.bigramcounts[bigram]/self.total_sentences

        return self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]

    def raw_unigram_probability(self, unigram):
        if unigram not in self.unigramcounts:
            return 1/self.total_words

        return self.unigramcounts[unigram]/self.total_words

    def smoothed_trigram_probability(self, trigram):
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return (lambda1 * self.raw_trigram_probability(trigram) +
                lambda2 * self.raw_bigram_probability(tuple(trigram[1:])) +
                lambda3 * self.raw_unigram_probability(tuple(trigram[2])))

    def sentence_logprob(self, sentence):
        ngrams = get_ngrams(sentence, 3)
        prob = 0
        for trigram in ngrams:
            prob += math.log2(self.smoothed_trigram_probability(trigram))
        return prob

    def perplexity(self, corpus):
        total_words_test = 0
        prob_sum_sentences = 0
        for test_sentence in corpus:
            prob_sum_sentences += self.sentence_logprob(test_sentence)
            total_words_test += len(test_sentence) + 1
        return 2 ** -(prob_sum_sentences/total_words_test)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0

        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp1 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp < pp1:
                correct += 1
            total += 1

        for f in os.listdir(testdir2):
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp3 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp2 < pp3:
                correct += 1
            total += 1

        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment:
    acc = essay_scoring_experiment('TOEFL_data/ets_toefl_data/train_high.txt',
                                   'TOEFL_data/ets_toefl_data/train_low.txt',
                                   'TOEFL_data/ets_toefl_data/test_high',
                                   'TOEFL_data/ets_toefl_data/test_low')
    print(acc)


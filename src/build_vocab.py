from collections import defaultdict
import pickle as pkl
import numpy as np
from gensim.models import KeyedVectors
from updated_normalization.normalization import normalization
import os


class Bilingual_preprocess():

    def __init__(self, src_language, en_preprocessed_bitext_path, src_preprocessed_bitext_path,
                 en_embedding_path, src_embedding_path,
                 output_dir, vocab_min_count):
        self.en_corpus_dir, self.src_corpus_dir = en_preprocessed_bitext_path, src_preprocessed_bitext_path
        self.en_embedding_path, self.src_embedding_path = en_embedding_path, src_embedding_path
        self.src_language = src_language
        self.output_dir = output_dir
        self.vocab_min_count = vocab_min_count


    def load_corpus_dataset(self):
        en_sentences, src_sentences = [], []
        with open(self.en_corpus_dir, 'r') as en_text_f:
            with open(self.src_corpus_dir, 'r') as src_text_f:
                en_line = en_text_f.readline()
                src_line = src_text_f.readline()
                while len(en_line) != 0 and len(src_line) != 0:
                    en_sentences.append(en_line.strip())
                    src_sentences.append(src_line.strip())
                    en_line = en_text_f.readline()
                    src_line = src_text_f.readline()
        self.en_sentences, self.src_sentences = en_sentences, src_sentences


    def preprocess_both_languages(self):
        self.load_corpus_dataset()

        # Tokenize the corpus of each language
        en_monolingual_preprocessor = Monolingual_preprocess(self.en_sentences, language='en',
                                                             embedding_path=self.en_embedding_path)
        en_monolingual_preprocessor.tokenize()
        src_monolingual_preprocessor = Monolingual_preprocess(self.src_sentences, language=self.src_language,
                                                              embedding_path=self.src_embedding_path)
        src_monolingual_preprocessor.tokenize()

        # Filter out the empty sentences
        nonempty_sent_idx = []
        for idx in range(len(en_monolingual_preprocessor.tokenized_corpus)):
            if len(en_monolingual_preprocessor.tokenized_corpus[idx]) != 0 and len(src_monolingual_preprocessor.tokenized_corpus[idx]) != 0:
                nonempty_sent_idx.append(idx)
        nonempty_en_tokenized_sentences = [en_monolingual_preprocessor.tokenized_corpus[idx] for idx in nonempty_sent_idx]
        nonempty_src_tokenized_sentences = [src_monolingual_preprocessor.tokenized_corpus[idx] for idx in nonempty_sent_idx]
        en_monolingual_preprocessor.tokenized_corpus = nonempty_en_tokenized_sentences
        src_monolingual_preprocessor.tokenized_corpus = nonempty_src_tokenized_sentences

        # Build vocab, vectorize, load corresponding embedding
        monolingual_preprocessors = {'en': en_monolingual_preprocessor, 'src': src_monolingual_preprocessor}
        for language in ['en', 'src']:
            monolingual_preprocessor = monolingual_preprocessors[language]
            monolingual_preprocessor.build_vocab(min_count=self.vocab_min_count)
            monolingual_preprocessor.vectorize_tokenized_sentences()
            monolingual_preprocessor.load_embedding_file()
            found_words, num_total_words = monolingual_preprocessor.create_embedding_matrix()

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            pkl.dump(monolingual_preprocessor.tokenized_corpus,
                     open(self.output_dir + 'tokenized_%s.pkl' % language, 'wb'))
            pkl.dump(monolingual_preprocessor.tokenized_int_arrs,
                     open(self.output_dir + 'int_arr_%s.pkl' % language, 'wb'))
            pkl.dump(monolingual_preprocessor.vocab,
                     open(self.output_dir + '%s_vocab.pkl' % language, 'wb'))
            pkl.dump(monolingual_preprocessor.id2word,
                     open(self.output_dir + '%s_id2word.pkl' % language, 'wb'))
            np.savetxt(self.output_dir + '%s_embed.npy' % language, monolingual_preprocessor.embedding)

            with open(self.output_dir + 'README', 'a') as f:
                f.write('%s: found %d words, total %d words.\n' % (language, found_words, num_total_words))



class Monolingual_preprocess():
    def __init__(self, corpus_sentences, language, embedding_path):
        self.corpus_sentences = corpus_sentences
        self.language = language
        self.embedding_path = embedding_path


    def tokenize(self):
        self.tokenized_corpus = self.tokenize_helper(self.corpus_sentences)


    def tokenize_helper(self, corpus_sentences):
        """
        Lowercase all words after tokenization.
        All stopwords/punctuations/newlines/consecutive spaces are removed.
        """
        tokenized_corpus = []
        for sentence in corpus_sentences:
            tokens = sentence.split()
            tokenized_sentence = []
            for token in tokens:
                if token.isspace() is False:
                    tokenized_sentence.append(token)
            tokenized_corpus.append(tokenized_sentence)

        return tokenized_corpus


    def build_vocab(self, min_count=3):
        """
        Create vocabulary set from the tokenized corpus.
        A vocabulary is included in the vocabulary set if it appears at least min_count number of times in the parallel
        corpus.
        """
        word2count = defaultdict(int)
        for sentence in self.tokenized_corpus:
            for word in sentence:
                word2count[word] += 1

        word2dict = {}
        word2dict['PAD'] = {'id': 0}
        word2dict['UNK'] = {'id': 1}
        for word in word2count:
            if word2count[word] >= min_count:
                word2dict[word] = {'id': len(word2dict), 'count': word2count[word]}
        self.vocab = word2dict


    def vectorize_tokenized_sentences(self):
        tokenized_int_arrs = []
        for sent in self.tokenized_corpus:
            int_arr = []
            for token in sent:
                if token in self.vocab:
                    int_arr.append(self.vocab[token]['id'])
                else:
                    int_arr.append(1)  # UNK token
            tokenized_int_arrs.append(int_arr)
        self.tokenized_int_arrs = tokenized_int_arrs


    def load_embedding_file(self):
        """
        Load the English/Lithuanian word embedding file in its correct format.
        Here we use Google News embedding for English and Fast Text embedding for Lithuanian.
        """
        if self.language == 'en':
            embed_file_dir = self.embedding_path
            wv = KeyedVectors.load_word2vec_format(embed_file_dir, binary=True)
            self.pretrained_embedding = {}
            for word in wv.vocab.keys():
                normalized_word = normalization.process(self.language.upper(), word, letters_to_keep='', letters_to_remove='',
                                                        lowercase=True, remove_repetitions_count=-1, remove_punct=True,
                                                        remove_digits=True, remove_vowels=False, remove_diacritics=True,
                                                        remove_spaces=False, remove_apostrophe=True, copy_through=False,
                                                        keep_romanized_text=False)
                self.pretrained_embedding[normalized_word] = wv[word]
            self.embed_dim = 300

        else:
            embed_file_dir = self.embedding_path
            fin = open(embed_file_dir, 'r', encoding='utf-8', newline='\n', errors='ignore')
            data = {}
            for line in fin:
                if len(line.split()) == 2:  # header
                    continue
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                normalized_word = normalization.process(self.language.upper(), word, letters_to_keep='', letters_to_remove='',
                                                        lowercase=True, remove_repetitions_count=-1, remove_punct=True,
                                                        remove_digits=True, remove_vowels=False, remove_diacritics=True,
                                                        remove_spaces=False, remove_apostrophe=True, copy_through=False,
                                                        keep_romanized_text=False)
                data[normalized_word] = np.array(tokens[1:])
            self.pretrained_embedding = data
            self.embed_dim = 300


    def create_embedding_matrix(self):
        """
        Create the embedding matrix for the created vocabulary set using the GoogleNews/FastText embedding.
        Words in the vocabulary set but not covered by the GoogleNews/FastText embedding are initialized with average word
        embedding of known words in the vocabulary set.
        UNK token is also initialized with average word embedding.
        """
        self.id2word = dict([(self.vocab[word]['id'], word) for word in self.vocab])
        vocab_size = len(self.vocab)
        result = np.zeros((vocab_size, self.embed_dim))
        unknown_token_set = set()

        found_words = 0
        avg = np.zeros(self.embed_dim)
        for _ in range(1, vocab_size): # skip PAD embedding (initialize as zero embedding)
            try:
                result[_] = self.pretrained_embedding[self.id2word[_]]
                avg += result[_]
                found_words += 1
            except:
                unknown_token_set.add(_)

        avg /= found_words
        for _ in unknown_token_set:
            result[_] = avg
        self.embedding = result
        return found_words, len(self.id2word)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build vocabulary from bitext.')
    parser.add_argument('--src_language', type=str, help='the low-resource language working with.')
    parser.add_argument('--en_preprocessed_bitext_path', action='store', type=str, help='input path of the preprocessed English bitext.')
    parser.add_argument('--src_preprocessed_bitext_path', action='store', type=str, help='input path of the preprocessed src-language bitext.')
    parser.add_argument('--en_embedding_path', action='store', type=str, help='input path of English word embeddings.')
    parser.add_argument('--src_embedding_path', action='store', type=str,
                        help='input path of src-language word embeddings.')
    parser.add_argument('--vocab_min_count', action='store', type=int,
                        help='the minimum number of times a word appears in the bitext in order for it to be included in the vocabulary.')
    parser.add_argument('--output_dir', action='store', type=str,
                        help='output directory of the vocabulary set and vectorized bitext.')

    args = parser.parse_args()
    preprocessor = Bilingual_preprocess(src_language=args.src_language,
                                        en_preprocessed_bitext_path=args.en_preprocessed_bitext_path,
                                        src_preprocessed_bitext_path=args.src_preprocessed_bitext_path,
                                        en_embedding_path=args.en_embedding_path,
                                        src_embedding_path=args.src_embedding_path,
                                        vocab_min_count=args.vocab_min_count,
                                        output_dir=args.output_dir)
    preprocessor.preprocess_both_languages()
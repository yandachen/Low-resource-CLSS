import re
from nltk.corpus import stopwords
import pickle as pkl
from model_def import NN_architecture
from model_wrapper import Model
import numpy as np
from updated_normalization.normalization import normalization
from updated_normalization.tokenization.mosestokenizer.src.mosestokenizer import *


class TokenizerNormalizer():
    def __init__(self, lang):
        self.lang = lang.upper()
        self.tokenizer = MosesTokenizer(lang.upper())
        self.punct_norm = MosesPunctuationNormalizer(lang.upper())

    def tokenize_normalize(self, line):
        line = line.strip()
        line = " ".join(self.tokenizer(self.punct_norm(line)))
        line = normalization.process(self.lang, line, letters_to_keep='', letters_to_remove='',
                                     lowercase=True, remove_repetitions_count=-1, remove_punct=True,
                                     remove_digits=True, remove_vowels=False, remove_diacritics=True,
                                     remove_spaces=False, remove_apostrophe=True, copy_through=False,
                                     keep_romanized_text=False)
        return line


class Sent_Rel_Predictor():
    def __init__(self, language, en_vocab_dir, src_vocab_dir,
                 model_architecture_kwargs, model_dir,
                 device):
        self.src_language = language
        self.en_vocab, self.src_vocab = pkl.load(open(en_vocab_dir, 'rb')), pkl.load(open(src_vocab_dir, 'rb'))
        self.model = NN_architecture(**model_architecture_kwargs)
        self.model_wrapper = Model(model=self.model, device=device, mode='eval', weight_dir=model_dir,
                                   batch_size=2048)

        # Load preprocessors
        self.load_preprocessors()


    def load_preprocessors(self):
        self.query_tokenizer_normalizer = TokenizerNormalizer(lang='en')
        self.sent_tokenizer_normalizer = TokenizerNormalizer(lang=self.src_language)

        # load stopwords
        stop_words = set(stopwords.words('english'))
        self.en_stopwords = set()
        for stop_word in stop_words:
            cleaned_word = self.query_tokenizer_normalizer.tokenize_normalize(stop_word)
            self.en_stopwords.add(cleaned_word)

        stop_words = set()
        with open('../stopwords/stopwords-%s.txt' % self.src_language, 'r') as f:
            stopwords_line = f.readline().strip()
            while len(stopwords_line) != 0:
                stop_words.add(stopwords_line.split('\n')[0])
                stopwords_line = f.readline().strip()
        self.src_stopwords = set()
        for stop_word in stop_words:
            cleaned_word = self.sent_tokenizer_normalizer.tokenize_normalize(stop_word)
            self.src_stopwords.add(cleaned_word)


    def predict(self, en_query, src_sentence):
        # en_query is a string of word/words, src_sentence is a string of word/words, where multiple queries are concatenated with ,
        # language is the src language, i.e. low-resource language

        # Preprocess source sentence
        clean_tokenized_line = self.sent_tokenizer_normalizer.tokenize_normalize(src_sentence).strip().split()
        clean_tokenized_line_without_stopwords = []
        for word in clean_tokenized_line:
            if word not in self.src_stopwords and word.isspace() is False:
                clean_tokenized_line_without_stopwords.append(word)

        # Preprocess English query
        queries = en_query.split(",")
        preprocessed_queries = []

        for q in queries:
            while '[' in q:
                left_bracket_idx = q.index('[')
                right_bracket_idx = q.index(']')
                q = q[:left_bracket_idx] + q[right_bracket_idx+1:]

            #remove all <> brackets grammatical constraints
            q = re.sub(r'[<>]', '', q)

            #remove all quotations marks and + sign
            q = re.sub(r'["+]', '', q)

            #remove all "EXAMPLE_OF" but keep the word inside the EXAMPLE_OF constraint
            while "EXAMPLE_OF" in q:
                example_of_start_idx = q.index('EXAMPLE_OF')
                left_bracket_idx = q.index('(')
                right_bracket_idx = q.index(')')
                q = q[:example_of_start_idx] + q[left_bracket_idx + 1: right_bracket_idx] + q[right_bracket_idx + 1:]
            preprocessed_queries.append(q)

        clean_tokenized_query_parts = []
        for query_part in preprocessed_queries:
            clean_tokenized_query_part = self.query_tokenizer_normalizer.tokenize_normalize(query_part).strip().split()
            # remove stopwords
            clean_tokenized_query_part_without_stopwords = []
            for word in clean_tokenized_query_part:
                if word not in self.en_stopwords and word.isspace() is False:
                    clean_tokenized_query_part_without_stopwords.append(word)
            if len(clean_tokenized_query_part_without_stopwords) != 0:
                clean_tokenized_query_parts.append(clean_tokenized_query_part_without_stopwords)

        # Replace word with word id
        vectorized_query_parts = []
        for query_part in clean_tokenized_query_parts:
            vectorized_query_part = []
            for word in query_part:
                if word in self.en_vocab:
                    vectorized_query_part.append(self.en_vocab[word]['id'])
                else:
                    vectorized_query_part.append(self.en_vocab['UNK']['id'])
            vectorized_query_parts.append(vectorized_query_part)

        vectorized_passage = []
        for word in clean_tokenized_line_without_stopwords:
            if word in self.src_vocab:
                vectorized_passage.append(self.src_vocab[word]['id'])
            else:
                vectorized_passage.append(self.src_vocab['UNK']['id'])

        # Predict with model
        features = []
        for query in vectorized_query_parts:
            if len(vectorized_passage) != 0 and len(query) != 0: # src sentence and query both contain at least one meaningful token
                features.append({'query': query, 'src': vectorized_passage})

        if len(features) == 0:
            # no feature in the query/sentence is recognized by the model as valid
            return 0
        relevant_scores = self.model_wrapper.predict(features)
        return np.min(relevant_scores) # take minimum across query parts


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Augment a relevance dataset based on parallel bitext.')
    parser.add_argument('--src_language', action='store', type=str,
                        help='the low-resource language working with.')
    parser.add_argument('--vocab_dir', action='store', type=str,
                        help='directory of the vocabulary set and the loaded word embeddings.')
    parser.add_argument('--model_path', action='store', type=str,
                        help='path of the trained SECLR-RT model.')
    parser.add_argument('--device', action='store', type=str,
                        help='device for training, e.g., "cuda:0"')

    args = parser.parse_args()

    model_architecture_kwargs = {'en_embedding_matrix': np.loadtxt(args.vocab_dir + 'en_embed.npy'),
                                 'src_embedding_matrix': np.loadtxt(args.vocab_dir + 'src_embed.npy'),
                                 }
    predictor = Sent_Rel_Predictor(language=args.src_language,
                                   en_vocab_dir=args.vocab_dir + 'en_vocab.pkl', src_vocab_dir=args.vocab_dir + 'src_vocab.pkl',
                                   model_architecture_kwargs=model_architecture_kwargs,
                                   model_dir=args.model_path,
                                   device=args.device)

    while True:
        en_query = input('En query:')
        src_sentence = input('Src sentence:')
        score = predictor.predict(en_query=en_query, src_sentence=src_sentence)
        print(score)

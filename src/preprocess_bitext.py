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


def preprocess_tokenize_raw_texts(src_language, en_bitext_path, src_bitext_path, en_stopwords_path, src_stopwords_path,
                                  output_preprocessed_en_bitext_path, output_preprocessed_src_bitext_path):
    bitext_paths = {'en': en_bitext_path, src_language: src_bitext_path}
    stopwords_paths = {'en': en_stopwords_path, src_language: src_stopwords_path}
    output_paths = {'en': output_preprocessed_en_bitext_path, src_language: output_preprocessed_src_bitext_path}
    for language in ['en', src_language]:
        # load stopwords
        stop_words = set()
        with open(stopwords_paths[language], 'r') as f:
            stopwords_line = f.readline().strip()
            while len(stopwords_line) != 0:
                stop_words.add(stopwords_line.split('\n')[0])
                stopwords_line = f.readline().strip()

        # preprocess the stopwords
        tokenizer_normalizer = TokenizerNormalizer(lang=language)
        preprocessed_stop_words = set()
        for stop_word in stop_words:
            preprocessed_word = tokenizer_normalizer.tokenize_normalize(stop_word)
            preprocessed_stop_words.add(preprocessed_word)

        # load the bitext, preprocess the bitext, and remove the stopwords
        tokenized_lines_without_stopwords = []
        with open(bitext_paths[language], 'r') as f:
            line = f.readline().strip()
            while len(line) != 0:
                preprocessed_line = tokenizer_normalizer.tokenize_normalize(line)
                tokenized_line = preprocessed_line.strip().split()
                tokenized_line_without_stopwords = []
                for token in tokenized_line:
                    if token not in preprocessed_stop_words:
                        tokenized_line_without_stopwords.append(token)
                tokenized_lines_without_stopwords.append(tokenized_line_without_stopwords)
                line = f.readline().strip()

        lines = [' '.join(line) for line in tokenized_lines_without_stopwords]
        with open(output_paths[language], 'w') as f:
            f.write('\n'.join(lines))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parse bitext / stopwords paths.')
    parser.add_argument('--src_language', type=str, help='the low-resource language working with.')
    parser.add_argument('--en_bitext_path', action='store', type=str, help='input path of the English bitext.')
    parser.add_argument('--src_bitext_path', action='store', type=str, help='input path of the src-language bitext.')
    parser.add_argument('--en_stopwords_path', action='store', type=str, help='input path of the English stopwords.')
    parser.add_argument('--src_stopwords_path', action='store', type=str, help='input path of the src-language stopwords.')
    parser.add_argument('--output_preprocessed_en_bitext_path', action='store', type=str,
                        help='output path of the preprocessed English bitext.')
    parser.add_argument('--output_preprocessed_src_bitext_path', action='store', type=str,
                        help='output path of the preprocessed src-language bitext.')

    args = parser.parse_args()
    preprocess_tokenize_raw_texts(args.src_language,
                                  args.en_bitext_path, args.src_bitext_path,
                                  args.en_stopwords_path, args.src_stopwords_path,
                                  args.output_preprocessed_en_bitext_path, args.output_preprocessed_src_bitext_path)
import numpy as np
import json
from cleaning_tok_transl import Cleaning


class PSQ():
    def __init__(self, psq_filename, cleaning_language=None):
        # load PSQ translation matrix
        self.translation_matrix = json.load(open(psq_filename, 'r'))
        self.query_not_exist_count = 0
        self.translation_not_exist_count = 0
        self.cleaning = None

        if cleaning_language is not None:
            self.cleaning_language = cleaning_language
            self.cleaned_lt_word_dict = {}
            self.cleaning = Cleaning(cleaning_language)


    def clean_word(self, word):
        if word in self.cleaned_lt_word_dict:
            return self.cleaned_lt_word_dict[word]
        else:
            if self.cleaning_language in ['lt', 'sw', 'so', 'tl', 'ps']:
                cleaned_word = self.cleaning.normalize(word, remove_repetitions=False, remove_vowels=False,
                                                   remove_spaces=False, remove_diacritics=True, transliterate=False)
            elif self.cleaning_language == 'bg':
                cleaned_word = self.cleaning.normalize(word, remove_repetitions=False, remove_vowels=False,
                                                      remove_spaces=False, remove_diacritics=True, transliterate=True)
            else:
                assert False
            self.cleaned_lt_word_dict[word] = cleaned_word
            return cleaned_word


    def compute_gt_rationale_distribution(self, query, src):
        assert len(query) == 1
        query_word = query[0]
        if query_word not in self.translation_matrix:
            self.query_not_exist_count += 1
            return None
        else:
            translation_probs = self.translation_matrix[query_word]
            if self.cleaning is not None:
                cleaned_src = [self.clean_word(src_word) for src_word in src]
            else:
                cleaned_src = src
            gt_rationale = []
            for lt_word in cleaned_src:
                if lt_word in translation_probs:
                    gt_rationale.append(translation_probs[lt_word])
                else:
                    gt_rationale.append(0)
            if np.sum(gt_rationale) == 0:
                self.translation_not_exist_count += 1
                return None
            else:
                normalize_sum = np.sum(gt_rationale)
                normalized_gt_rationale = gt_rationale / normalize_sum
                return normalized_gt_rationale

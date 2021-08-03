import normalization
import sys
from mosestokenizer import *

if __name__=="__main__":
    lang = sys.argv[1]
    tokenizer = MosesTokenizer(lang)
    punct_norm = MosesPunctuationNormalizer(lang)    
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        line = " ".join(tokenizer(punct_norm(line)))
        normalized_text = normalization.process(lang, 
                                                line, 
                                                letters_to_keep='', 
                                                letters_to_remove='', 
                                                lowercase=True, 
                                                remove_repetitions_count=-1, 
                                                remove_punct=True,
                                                remove_digits=True,
                                                remove_vowels=False,
                                                remove_diacritics=True,
                                                remove_spaces=False,
                                                remove_apostrophe=True,
                                                copy_through=False,
                                                keep_romanized_text=False)
        print(line)

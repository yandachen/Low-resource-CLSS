from mosestokenizer import *
import SCRIPTS_Normalization.normalization as normalization

class Cleaning:
    
    def __init__(self,language):
        self.tokenize = MosesTokenizer(language)
        self.norm = MosesPunctuationNormalizer(language)
        self.language = language

    def normalize(self, text, norm_dict):
        if text: text = " ".join(self.tokenize(self.norm(text)))
        
        copy_through = False
        keep_romanized_text = False
        
        if norm_dict != None:
            copy_through=norm_dict['copy_through']
            keep_romanized_text=norm_dict['keep_romanized_text']
        
        return normalization.process(self.language, text, copy_through=copy_through, keep_romanized_text=keep_romanized_text)


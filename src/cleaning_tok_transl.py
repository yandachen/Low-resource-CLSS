# coding=utf-8
# -*- coding: utf-8 -*-

import re
import unicodedata
from mosestokenizer import *

class Cleaning:
    
    def __init__(self,language):
        self.language = language
        english_alphabet = "abcdefghijklmnopqrstuvwxyz"
        english_vowels = "aeiou"

        swahili_alphabet = "abcdefghijklmnopqrstuvwxyz"
        swahili_vowels = "aeiou"

        tagalog_alphabet = "abcdefghijklmnopqrstuvwxyz"
        tagalog_vowels = "aeiou"

        somali_alphabet = "abcdefghijklmnopqrstuvwxyz"
        somali_vowels = "aeiou"

        lithuanian_alphabet = "aąbcčdeęėfghiįyjklmnoprsštuųūvzž"
        lithuanian_vowels = "aąeęėiįouųū"

        bulgarian_alphabet = "абвгдежзийклмнопрстуфхцчшщъьюя"
        bulgarian_vowels = "аеиоуъ"

        pashto_alphabet = "واهدرنلیيېمکتپسبخړشغچزګفځعښټډحڅجقږصۍژطكئضىظڼثذآ۶گہؤےءةأھۀإﺉًٌٍَُِّْ"
        pashto_vowels = "واېيىیےۍ"
        
        self.alphabet_map = {}
        self.alphabet_map["ENG"] = english_alphabet;
        self.alphabet_map["TGL"] = tagalog_alphabet;
        self.alphabet_map["SWA"] = swahili_alphabet;
        self.alphabet_map["SOM"] = somali_alphabet;
        self.alphabet_map["LIT"] = lithuanian_alphabet;
        self.alphabet_map["BUL"] = bulgarian_alphabet;
        self.alphabet_map["PUS"] = pashto_alphabet;
        
        self.vowels_map = {}
        self.vowels_map["ENG"] = english_vowels;
        self.vowels_map["TGL"] = tagalog_vowels;
        self.vowels_map["SWA"] = swahili_vowels;
        self.vowels_map["SOM"] = somali_vowels;
        self.vowels_map["LIT"] = lithuanian_vowels;
        self.vowels_map["BUL"] = bulgarian_vowels;
        self.vowels_map["PUS"] = pashto_vowels;

        ''' Special cxases not handled by the default unicode undiacritization '''
        self.latin_character_mappings = {
            'ą': 'a',
            'č': 'c',
            'ę': 'e',
            'ė': 'e',
            'į': 'i',
            'š': 's',
            'ų': 'u',
            'ū': 'u',
            'ž': 'z',
        }
        
        ''' Special transformation for Pashto '''
        self.pashto_character_mappings = {
            'ق': 'ک',
            'ف': 'پ',
            'ك': 'ک',
            'گ': 'ګ',
            'ﺉ': 'ئ',
            'ء': '۶',
            'ہ': 'ه',
            'ھ': 'ه',
            '۵': '٥',
            '۴': '٤',
                'ے': 'ي',
            'ۍ': 'ي',
            'ی': 'ي',
            'ى': 'ي',
            'ې': 'ي',
            'إ': 'ا',
            'آ': 'ا',
            'أ': 'ا',
            'ة': 'ه',
        }

        self.bulgarian_transliteration = {
            'ch': 'ч',
            'ja': 'я',
            'ju': 'ю',
            'kh': 'х',
            'sht': 'щ',
            'sh': 'ш',
            'ya': 'я',
            'yu': 'ю',
            'zh': 'ж',
            'a': 'а',
            'b': 'б',
            'c': 'ц',
            'd': 'д',
            'e': 'е',
            'f': 'ф',
            'g': 'г',
            'h': 'х',
            'i': 'и',
            'j': 'й',
            'k': 'к',
            'l': 'л',
            'm': 'м',
            'n': 'н',
            'o': 'о',
            'p': 'п',
            'r': 'р',
            's': 'с',
            't': 'т',
            'u': 'у',
            'v': 'в',
            'x': 'х',
            'y': 'й',
            'z': 'з',   
        }
        self.tokenize = MosesTokenizer(language)
        self.norm = MosesPunctuationNormalizer(language)

    def normalize(self, text, remove_repetitions, remove_vowels, remove_spaces, remove_diacritics=True, transliterate=True):
        remove_repetitions_count = 0
        if remove_repetitions == True:
            remove_repetitions_count = 1
            
        return self.normalize_full(text, "", "", True, remove_repetitions_count, True, True, remove_vowels, remove_diacritics, remove_spaces, True, transliterate)
        
    def normalize_full(self, text, letters_to_keep, letters_to_remove, lowercase, remove_repetitions_count, remove_punct, remove_digits, remove_vowels, remove_diacritics, remove_spaces, remove_apostrophe, transliterate):
        '''
        Normalization and cleaning-up text
        '''

        if text: text = " ".join(self.tokenize(self.norm(text)))
        
        language = self.language.upper()
        if (language == 'ENGLISH') or (language == 'ENG') or (language == 'EN'):
            language = "ENG"
        elif (language == '1A') or (language == 'SWAHILI') or (language == 'SWA') or (language == 'SW'):
            language = "SWA"
        elif (language == '1B') or (language == 'TAGALOG') or (language == 'TGL') or (language == 'TL'):
            language = "TGL"
        elif (language == '1S') or (language == 'SOMALI') or (language == 'SOM') or (language == 'SO'):
            language = "SOM"
        elif (language == '2B') or (language == 'LITHUANIAN') or (language == 'LIT') or (language == 'LT'):
            language = "LIT"
        elif (language == '2S') or (language == 'BULGARIAN') or (language == 'BUL') or (language == 'BG'):
            language = "BUL"
        elif (language == '2C') or (language == 'PASHTO') or (language == 'PUS') or (language == 'PS'):
            language = "PUS"

        alphabet = self.alphabet_map[language]
        vowels = self.vowels_map[language]
        pashto_diacs = "ًَ"
        
        if language == "BUL":
            if transliterate == True:
                for key in self.bulgarian_transliteration:
                    text = re.sub(r''+key, self.bulgarian_transliteration[key], text)
                    text = re.sub(r''+key.upper(), self.bulgarian_transliteration[key].upper(), text)

        '''Prepare the lists of the letters to be explictily kept and removed'''
        letters_in = list(letters_to_keep)
        letters_out = list(letters_to_remove)

        '''Lowercase text, if required'''
        if lowercase == True:
            text = text.lower()

        '''Remove repititions of a specific length, if required'''
        if remove_repetitions_count > 0:
            replacement = r''
            for count in range(remove_repetitions_count):
                replacement += '\\1'
            text = re.sub(r'(.)\1{'+str(remove_repetitions_count)+',}', replacement, text)

        '''Remove punctuation marks, if required'''
        if remove_punct == True:
            text = re.sub(r"[^\w\s\']",'', text)
            text = re.sub(r"(^|\s)[\']", r'\1', text) 

        '''Remove digits, if required'''
        if remove_digits == True:
            text = re.sub(r'\d', '', text)

        '''Remove apostrophe, if required'''
        if remove_apostrophe == True:
            text = re.sub(r'\'', '', text)

        '''Remove spaces, if required.''' 
        if remove_spaces == True:
            text = re.sub(r'\s', '', text)

        '''Loop over the unique characters in the text'''
        for char in list(set(text)):
            if not char.isspace() and not char.isdigit() and not re.match(r"[^\w\s\d]", char):

                '''If the character is needed to be removed, remove it'''
                if char in letters_out:
                    text = re.sub(re.escape(char), '', text)
                    continue

                '''Remove diacritics, if required.'''
                if char not in letters_in and remove_diacritics:
                    lower = char == char.lower()
                    char_norm = char
                    if language != "BUL" and char.lower() in self.latin_character_mappings:
                        char_norm = self.latin_character_mappings[char.lower()]
                    if language == "PUS" and char.lower() in pashto_diacs:
                        char_norm = ''
                    elif char.lower() not in alphabet:
                        char_norm = unicodedata.normalize('NFD', char)
                        char_norm = char_norm.encode('ascii', 'ignore')
                        char_norm = char_norm.decode("utf-8")
                    if not lower:
                        char_norm = char_norm.upper()
                    if char != char_norm:
                        text = re.sub(re.escape(char), char_norm, text)
                        char = char_norm

                ''' Remove any character that is not in the alphabet. Also, remove vowels, if required '''
                if char not in letters_in and (char in letters_out or (char.lower() in vowels and remove_vowels == True) or char.lower() not in alphabet) and (language != "BUL" or transliterate == False):
                    text = re.sub(re.escape(char), '', text)

        '''Remove extra spaces'''
        text = re.sub(r'\s+', ' ', text).strip()

        return text
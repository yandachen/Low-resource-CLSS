'''
Updated on August 17th, 2020, 4:40pm
Version 2.2b

@author: reskander
'''
# coding=utf-8

import sys
import re
import unicodedata

english_alphabet = "abcdefghijklmnopqrstuvwxyz"
english_vowels = "aeiou"

swahili_alphabet = "abcdefghijklmnopqrstuvwxyz"
swahili_vowels = "aeiou"

tagalog_alphabet = "abcdefghijklmnopqrstuvwxyz"
tagalog_vowels = "aeiou"

somali_alphabet = "abcdefghijklmnopqrstuvwxyz'"
somali_vowels = "aeiou"

lithuanian_alphabet = "aąbcčdeęėfghiįyjklmnoprsštuųūvzž"
lithuanian_vowels = "aąeęėiįouųū"

bulgarian_alphabet = "абвгдежзийклмнопрстуфхцчшщъьюяѫѣ"
bulgarian_romanized_alphabet = "ŭĭu̐abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъьюяѫѣ"
bulgarian_vowels = "аеиоуъ"

pashto_alphabet = "واهدرنلیيېمکتپسبخړشغچزګفځعښټډحڅجقږصۍژطكئضىظڼثذآ۶گہؤےءةأھۀإﺉًٌٍَُِّْ"
pashto_romanized_alphabet = "abcdefghijklmnopqrstuvwxyzواهدرنلیيېمکتپسبخړشغچزګفځعښټډحڅجقږصۍژطكئضىظڼثذآ۶گہؤےءةأھۀإﺉًٌٍَُِّْ"
pashto_vowels = "واېيىیےۍ"
pashto_diacs = "ًٌٍَُِّْ"

farsi_numbers = u'\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9'
farsi_alphabet = u'\u0621\u0622\u0623\u0624\u0626\u0627\u0628\u067e\u062a\u062b\u062c\u0686\u062d\u062e\u062f\u0630\u0631\u0632\u0698\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063a\u0641\u0642\u06a9\u06af\u0644\u0645\u0646\u0647\u0648\u06cc\u064b\u0654=' + farsi_numbers
farsi_romanized_alphabet = "abcdefghijklmnopqrstuvwxyz_'|^W}AJCH+$SDTZEQGF%_=" + farsi_alphabet
farsi_vowels = "‬ًًٌَُِّْﻭﺍېﻱﻯیےۍ" + u'\u0650\u064B\u064E\u064F\u0652\u064C\u064C\u064D'
farsi_diacs = pashto_diacs + u'\u0640\u0651'

alphabet_map = {}
alphabet_map["ENG"] = english_alphabet;
alphabet_map["TGL"] = tagalog_alphabet;
alphabet_map["SWA"] = swahili_alphabet;
alphabet_map["SOM"] = somali_alphabet;
alphabet_map["LIT"] = lithuanian_alphabet;
alphabet_map["BUL"] = bulgarian_alphabet;
alphabet_map["BUL_ROM"] = bulgarian_romanized_alphabet;
alphabet_map["PUS"] = pashto_alphabet;
alphabet_map["PUS_ROM"] = pashto_romanized_alphabet;
alphabet_map["FAS"] = farsi_alphabet;
alphabet_map["FAS_ROM"] = farsi_romanized_alphabet;

vowels_map = {}
vowels_map["ENG"] = english_vowels;
vowels_map["TGL"] = tagalog_vowels;
vowels_map["SWA"] = swahili_vowels;
vowels_map["SOM"] = somali_vowels;
vowels_map["LIT"] = lithuanian_vowels;
vowels_map["BUL"] = bulgarian_vowels;
vowels_map["PUS"] = pashto_vowels;
vowels_map["FAS"] = farsi_vowels;

''' Special cases not handled by the default unicode undiacritization '''
latin_character_mappings = {
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
pashto_character_mappings = {
    'ق': 'ک',
    'ف': 'پ',
    'ك': 'ک',
    'گ': 'ګ',
    'ﺉ': 'ئ',
    'ء': '6',
    'ہ': 'ه',
    'ھ': 'ه',
    '۵': '٥',
    '۴': '٤',
    'ٸ': 'ئ',
    'ؤ': 'و',
    'ﻻ': 'لا',
    'ۓ': 'ي',
    'ے': 'ي',
    'ﮮ': 'ي',
    'ۍ': 'ي',
    'ی': 'ي',
    'ى': 'ي',
    'ې': 'ي',
    'إ': 'ا',
    'آ': 'ا',
    'أ': 'ا',
    'ة': 'ه',
    'ۀ': 'ه',
}

farsi_character_mappings = {
   '=': u'\u200c',
   'آ': u'\u0627',
    u'\u0649': u'\u06cc',
    u'\u06BE': u'\u0647',
    u'\u06C0': u'\u0647',
    u'\u0629': u'\u0647',
    u'\u0625': u'\u0627',
    'ي': u'\u06CC',
    'ﺅ': u'\u0648',
    'ﺉ': "ی",
    'ﭘ': u'\u0641',
    'ﻕ': u'\u0641',
    'ﻑ': u'\u0641',
    'ﻙ': u'\u06a9',
    u'\u064A' : u'\u06CC',
    u'\u0643' : u'\u06A9',
    'ﺀ': u'\u06f6',
    'ہ': u'\u0647',
    'ھ': u'\u0647',
    '۵': u'\u06f5',
    '۴': u'\u06f4',
    'ٸ': u'\u06cc',
    'ﺅ': u'\u0648',
    'ﻻ': 'ﻻ',
    'ۓ': u'\u06CC',
    'ے': u'\u06CC',
    'ﮮ': u'\u06CC',
    'ۍ': u'\u06CC',
    'ی': u'\u06CC',
    'ﻯ': u'\u06CC',
    'ې': u'\u06CC',
    'ﺇ': u'\u0627',
    'ﺁ': u'\u0627',
    'ﺃ': u'\u0627',
    'ﺓ': u'\u0647',
    'ۀ': u'\u0647',
    u'\u0660': u'\u06f0',
    u'\u0661': u'\u06f1',
    u'\u0662': u'\u06f2',
    u'\u0663': u'\u06f3',
    u'\u0664': u'\u06f4',
    u'\u0665': u'\u06f5',
    u'\u0666': u'\u06f6',
    u'\u0667': u'\u06f7',
    u'\u0668': u'\u06f8',
    u'\u0669': u'\u06f9',
    '0': u'\u06f0',
    '1': u'\u06f1',
    '2': u'\u06f2',
    '3': u'\u06f3',
    '4': u'\u06f4',
    '5': u'\u06f5',
    '6': u'\u06f6',
    '7': u'\u06f7',
    '8': u'\u06f8',
    '9': u'\u06f9',
    u'\u2776': u'\u06f1',
    u'\uFE8E': u'\u0627',
    u'\uFEAE': u'\u0631',
    u'\uFE91': u'\u0628',
    u'\uFEE3': u'\u0645',
    u'\uFEEE': u'\u0648',
    u'\uFEAA': u'\u062f',
    u'\u2776': u'\u06f8',
    u'\uFEAA': u'\u0647',
    u'\uFEF3': u'\u06cc',
    u'\u06D5': u'\u0639',
    u'\uFEB7': u'\u0634',
    u'\uFEE7': u'\u0646',
    u'\uFEE8': u'\u0646',
    u'\uFEEB': u'\u0647',
    u'\uFEB3': u'\u0633',
    u'\uFEAD': u'\u0631',
    u'\uFEA9': u'\u062f',
    u'\uFEA7': u'\u062e',
    u'\uFE97': u'\u062a',
    u'\uFE8D': u'\u0627',
    u'\uFE92': u'\u0628',
    u'\uFE98': u'\u062a',
    u'\uFEE4': u'\u0645',
    u'\uFEB8': u'\u0634',
    u'\uFEDB': u'\u06a9',
    u'\u05D1': u'\u067e',
    u'\uFEAF': u'\u0632',
    u'\uFECB': u'\u0639',
    u'\uFEF4': u'\u06cc',
    u'\uFB94': u'\u06af',
    u'\uFBFD': u'\u06cc',
    u'\uFEE5': u'\u0646',
    u'\uFBFE': u'\u06cc',
    u'\uFEED': u'\u0648',
    u'\uFEEC': u'\u0647',
    u'\uFEB4': u'\u0633',
    u'\uFEDF': u'\u0644',
    u'\uFEF2': u'\u06cc',
    u'\uFEE1': u'\u0645',
    u'\uFBFF': u'\u06CC',
    u'\uFEB0': u'\u0632',
    u'\uFE9F': u'\u062C',
    u'\uFEF1': u'\u06CC',
    u'\uFEE9': u'\u0647',
    u'\uFEE6': u'\u0646',
    u'\uFB90': u'\u06A9',
    u'\ufee2': u'\u0645',
    u'\uFED8': u'\u0642',
    u'\uFEDE': u'\u0644',
    u'\uFECC': u'\u0639',
    u'\uFED3': u'\u0641',
    u'\u06AB': u'\u06af',
    u'\uFEE0': u'\u0644',
    u'\uFEB6': u'\u0634',
    u'\uFED7': u'\u0634',
    u'\u06AA': u'\u06af',
    u'\uFECF': u'\u063a',
    u'\uFECF': u'\u063a',
    u'\u0142': 'l',
    u'\uFEDC': u'\u06a9',
    u'\uFED4': u'\u0641',
    u'\u067c': u'\u062a',
    u'\u067c': u'\u062d',
    u'\uFEA0': u'\u062c',
    u'\uFEC2': u'\u0637',

}

#source: https://www.loc.gov/catdir/cpso/romanization/bulgarian.pdf
bulgarian_transliteration = {
    'ch': 'ч',
    'ja': 'я',
    'ju': 'ю',
    'kh': 'х',
    'sht': 'щ',
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
    'ŭ': 'Ъ',
    '′': 'ь',
    '″': 'ъ',
    'i͡e': 'ѣ',
    'i͡a': 'я',
    'i͡u': 'ю',
    'ĭ': 'й',
    'u̐': 'ѫ'
}

farsi_transliteration = {
    'ء': "'",
    'آ': "|",
    'أ': "^",
    'ؤ': "W",
    'ئ': "}",
    'ا': "A",
    'ب': "b",
    'پ': "p",
    'ت': "t",
    'ث': "v",
    'ج': "J",
    'چ': "C",
    'ح': "H",
    'خ': "x",
    'د': "d",
    'ذ': "+",
    'ر': "r",
    'ز': "z",
    'ژ': "c",
    'س': "s",
    'ش': "$",
    'ص': "S",
    'ض': "D",
    'ط': "T",
    'ظ': "Z",
    'ع': "E",
    'غ': "g",
    'ف': "f",
    'ق': "q",
    'ک': "Q",
    'گ': "G",
    'ل': "l",
    'م': "m",
    'ن': "n",
    'ه': "h",
    'و': "w",
    'ی': "y",
    'ً': "F",
    "'": "%",
    '_': "_",
#    '‌': "=",
}

def process(language, text, letters_to_keep='', letters_to_remove='', lowercase=True, remove_repetitions_count=-1, remove_punct=True, remove_digits=True, remove_vowels=False, remove_diacritics=True, remove_spaces=False, remove_apostrophe=True, copy_through=True, keep_romanized_text=True):
    '''
    Normalization and cleaning-up text
    '''
    alphabet = None
    vowels = None
    language = language.upper()
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
    elif (language == '3S') or (language == 'FARSI') or (language == 'FAS') or (language == 'FA'):
        language = "FAS"
        
    alphabet = alphabet_map[language]
    if language == 'BUL' and keep_romanized_text:
        alphabet = alphabet_map['BUL_ROM']
    if language == 'PUS' and keep_romanized_text:
        alphabet = alphabet_map['PUS_ROM']
    if language == 'FAS' and keep_romanized_text:
        alphabet = alphabet_map['FAS_ROM']
    vowels = vowels_map[language]

    if language == "BUL" and not keep_romanized_text:
        for key in bulgarian_transliteration:
            text = re.sub(r''+key, bulgarian_transliteration[key], text)
            text = re.sub(r''+key.upper(), bulgarian_transliteration[key].upper(), text)

    if language == "PUS":
        for key in pashto_character_mappings:
            text = re.sub(r''+key, pashto_character_mappings[key], text)

    if language == "FAS":
        for key in farsi_character_mappings:
            text = re.sub(r''+key, farsi_character_mappings[key], text)
  
    '''Prepare the lists of the letters to be explictily kept and removed'''
    letters_in = list(letters_to_keep)
    letters_out = list(letters_to_remove)
    
    '''Lower-case text, if required'''
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
        text = re.sub(r"[^\w\s\'\َ\ً\ُ\ِ\ْ\ّ\ٌ\ٍ۰]",'', text)
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
        #Special handling for zero-width non-joiner (do not replace)
        if (language == 'PUS' or language == "FAS") and ord(char) == 8204:
            continue
        if not char.isspace() and not char.isdigit() and not re.match(r"[^\w\s\'\َ\ً\ُ\ِ\ْ\ّ\ٌ\ٍ\d]", char):
            '''If the character is needed to be removed, remove it'''
            if char in letters_out:
                text = re.sub(re.escape(char), '', text)
                continue

            '''Remove vowels, if required.'''
            if char not in letters_in and remove_vowels:
                char_norm = char
                if char.lower() in vowels:
                    char_norm = ''
                if char != char_norm:
                    text = re.sub(re.escape(char), char_norm, text)
                    char = char_norm

            '''Remove diacritics, if required.'''
            if char not in letters_in and remove_diacritics:
                lower = char == char.lower()
                char_norm = char
                if char.lower() in latin_character_mappings:
                    char_norm = latin_character_mappings[char.lower()]
                elif char.lower() in pashto_diacs and language == "PUS":
                    char_norm = ''
                elif char.lower() in farsi_diacs and language == "FAS":
                    char_norm = ''
                elif char.lower() not in alphabet:
                    char_norm_nfd = unicodedata.normalize('NFD', char.lower())
                    char_norm_ascii = char_norm_nfd.encode('ascii', 'ignore')
                    char_norm_ascii = char_norm_ascii.decode("utf-8")
                    char_norm = char_norm_ascii
                    if len(char_norm) == 0:
                        char_norm = char_norm_nfd
                    if char_norm == ' ':
                        char_norm = char.lower()
                if not lower:
                    char_norm = char_norm.upper()
                if char != char_norm:
                    text = re.sub(re.escape(char), char_norm, text)
                    char = char_norm

            ''' Remove any character that is not in the alphabet, if otherwise specified'''
            if not copy_through and char not in letters_in and (char in letters_out or char.lower() not in alphabet):
                text = re.sub(re.escape(char), '', text)

    '''Remove extra spaces'''
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

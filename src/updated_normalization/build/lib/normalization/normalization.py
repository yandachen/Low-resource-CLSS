'''
Updated on April 21st, 2021
Version 4.0

@author: reskander
'''
# coding=utf-8

import unicodedata

from .constants import *

def process(language, text, letters_to_keep='', letters_to_remove='', lowercase=False, remove_repetitions_count=-1, remove_punct=False, remove_digits=False, remove_vowels=False, remove_diacritics=True, remove_spaces=False, remove_apostrophe=False, copy_through=True, keep_romanized_text=True):

    '''
    Normalization and cleaning-up text
    '''
    alphabet = None
    vowels = None
    language = language.upper()
    if language == 'ENGLISH' or language == 'ENG' or language == 'EN':
        language = "ENG"
    elif language == '1A' or language == 'SWAHILI' or language == 'SWA' or language == 'SW':
        language = "SWA"
    elif language == '1B' or language == 'TAGALOG' or language == 'TGL' or language == 'TL':
        language = "TGL"
    elif language == '1S' or language == 'SOMALI' or language == 'SOM' or language == 'SO':
        language = "SOM"
    elif language == '2B' or language == 'LITHUANIAN' or language == 'LIT' or language == 'LT':
        language = "LIT"
    elif language == '2S' or language == 'BULGARIAN' or language == 'BUL' or language == 'BG':
        language = "BUL"
    elif language == '2C' or language == 'PASHTO' or language == 'PUS' or language == 'PS':
        language = "PUS"
    elif language == '3S' or language == 'FARSI' or language == 'PERSIAN' or language == 'FAS' or language == 'PER' or language == 'FA':
        language = "FAS"
    elif language == '3C' or language == 'KAZAKH' or language == 'KAZ' or language == 'KK':
        language = "KAZ"
    elif language == '3B' or language == 'GEORGIAN' or language == 'KAT' or language == 'KA':
        language = "KAT"

    alphabet = alphabet_map[language]
    vowels = vowels_map[language]
    if language == 'BUL' and keep_romanized_text:
        alphabet = alphabet_map['BUL_ROM']
        vowels = vowels_map['BUL_ROM']
    if language == 'PUS' and keep_romanized_text:
        alphabet = alphabet_map['PUS_ROM']
        vowels = vowels_map['PUS_ROM']
    if language == 'FAS' and keep_romanized_text:
        alphabet = alphabet_map['FAS_ROM']
        vowels = vowels_map['FAS_ROM']
    if language == 'KAZ' and keep_romanized_text:
        alphabet = alphabet_map['KAZ_ROM']
        vowels = vowels_map['KAZ_ROM']
    if language == 'KAT' and keep_romanized_text:
        alphabet = alphabet_map['KAT_ROM']
        vowels = vowels_map['KAT_ROM']

    '''Prepare the lists of the letters to be explictily kept and removed'''
    letters_in = list(letters_to_keep)
    letters_out = list(letters_to_remove)


    '''Remove extras, e.g., non-zero width jopiner, and non-printable characters'''
    text = re.sub(extras, '', text)
    text = "".join([char for char in text if char.isprintable()])


    '''Transliteration for Bulgarian'''
    if language == "BUL" and not keep_romanized_text:
        for key in bulgarian_latin_transliteration:
            if key not in letters_in:
                text = re.sub(r''+key, bulgarian_latin_transliteration[key], text)
                text = re.sub(r''+key.upper(), bulgarian_latin_transliteration[key].upper(), text)


    '''Mapping for Pashto'''
    if language == "PUS":
        for key in pashto_character_mappings:
            if key not in letters_in:
                text = re.sub(r''+key, pashto_character_mappings[key], text)


    '''Mapping for Farsi'''
    if language == "FAS":
        for key in farsi_character_mappings:
            if key not in letters_in:
                text = re.sub(r''+key, farsi_character_mappings[key], text)


    '''Transliteration for Farsi'''
    if language == "FAS" and not keep_romanized_text:
        for key in farsi_latin_transliteration:
            if key not in letters_in:
                text = re.sub(r''+key, farsi_latin_transliteration[key], text)
                text = re.sub(r''+key.upper(), farsi_latin_transliteration[key].upper(), text)


    '''Mapping for kazakh''' #homoglyphs
    #Resolve homoglyphs if at least one character is Cyrillic, otherwise the homoglyph is likely to be intended.
    at_least_one_cyrillic_letter = '.*['+kazakh_alphabet+'].*'
    if language == "KAZ" and re.match(at_least_one_cyrillic_letter, text):
        for key in kazakh_character_mappings:
            if key not in letters_in:
                text = re.sub(r'' + key, kazakh_character_mappings[key], text)


    '''Transliteration for kazakh'''
    # Latin transliteration is conditioned on keep_romanized_text
    # Arabic transliteration is always forced
    # Kazakh Cyrillic to Cyrillic is not applied for now (the map is calculated though).
    if language == "KAZ" and not keep_romanized_text:
        for key in kazakh_latin_transliteration:
            if key not in letters_in:
                text = re.sub(r'' + key, kazakh_latin_transliteration[key], text)
                text = re.sub(r'' + key.upper(), kazakh_latin_transliteration[key].upper(), text)
    if language == "KAZ":
        #Remove Arabic diacs in all cases as the Arabic letters get automatically transliterated
        text = re.sub(kazakh_diac, '', text)
        for key in kazakh_arabic_transliteration:
            if key not in letters_in:
                text = re.sub(r'' + key, kazakh_arabic_transliteration[key], text)


    '''Transliteration for Georgian'''
    if language == "KAT" and not keep_romanized_text:
        for key in georgian_latin_transliteration:
            if key not in letters_in:
                text = re.sub(r''+key, georgian_latin_transliteration[key], text)
                text = re.sub(r''+key.upper(), georgian_latin_transliteration[key].upper(), text)

    '''Lower-case text, if required'''
    if lowercase:
        text = text.lower()


    '''Remove repetitions of a specific length, if required'''
    if remove_repetitions_count > 0:
        replacement = ''
        for count in range(remove_repetitions_count):
            replacement += '\\1'
        text = re.sub(r'(.)\1{'+str(remove_repetitions_count)+',}', replacement, text)


    '''Remove punctuation marks, if required'''
    if remove_punct:
        tokens = text.split()
        no_punc = []
        for token in tokens:
            if bool(re.match(number, token)):
                no_punc.append(token)
            elif not bool(re.match("^\'+$", token)):
                no_punc_token = ''
                for char in token:
                    if not is_punc(char) or char in letters_in:
                        no_punc_token += char
                if len(no_punc_token) > 0:
                    no_punc.append(no_punc_token)
        text = ' '.join(no_punc)


    '''Remove digits, if required'''
    if remove_digits:
        tokens = text.split()
        no_numbers = []
        for token in tokens:
            if not bool(re.match(number, token)):
                no_numbers.append(token)
        text = ' '.join(no_numbers)
        text = re.sub(digit, '', text)


    '''Remove apostrophe, if required'''
    if remove_apostrophe:
        tokens = text.split()
        no_apostrophes = []
        for token in tokens:
            if not bool(re.match("^[\'\ʼ]+$", token)):
                token = re.sub('[\'\ʼ]', '', token)
            no_apostrophes.append(token)
        text = ' '.join(no_apostrophes)


    '''Remove spaces, if required.'''
    if remove_spaces:
        text = "".join([char for char in text if not char.isspace()])


    '''Loop over the unique characters in the text'''
    for char in list(set(text)):
        #Special handling for zero-width non-joiner (do not replace)
        #if (language == 'PUS' or language == "FAS") and ord(char) == 8204:
        #    continue
        if (not char.isspace() and char.isprintable() and not is_punc(char) and not bool(re.match(digit, char))) or char in alphabet:
            char_lower = char.lower()
            '''If the character is needed to be removed, remove it'''
            if char in letters_out:
                text = re.sub(re.escape(char), '', text)
                continue

            '''Remove diacritics, if required.'''
            if char not in letters_in and remove_diacritics:
                lower = (char == char.lower())
                char_norm = char.lower()
                if char_lower in diac_character_mappings:
                    char_norm = diac_character_mappings[char_lower]
                else:
                    char_norm_nfd = unicodedata.normalize('NFD', char_lower)
                    char_norm_ascii = char_norm_nfd.encode('ascii', 'ignore')
                    char_norm_ascii = char_norm_ascii.decode("utf-8")
                    char_norm = char_norm_ascii
                    if len(char_norm) == 0:
                        char_norm = char_norm_nfd
                    if char_norm == ' ':
                        char_norm = char_lower

                # After removing the diacritics, some characters might need to be removed or transliterated.
                if language == 'BUL' and not keep_romanized_text and char_norm in bulgarian_latin_transliteration:
                    char_norm = bulgarian_latin_transliteration[char_norm]
                elif language == 'PUS' and char_norm in pashto_diac:
                    char_norm = ''
                elif language == 'FAS' and char_norm in farsi_diac:
                    char_norm = ''
                elif language == 'FAS' and not keep_romanized_text and char_norm in farsi_latin_transliteration:
                    char_norm = farsi_latin_transliteration[char_norm]
                elif language == 'KAZ' and not keep_romanized_text and char_norm in kazakh_latin_transliteration:
                    char_norm = kazakh_latin_transliteration[char_norm]
                elif language == 'KAT' and not keep_romanized_text and char_norm in georgian_latin_transliteration:
                    char_norm = georgian_latin_transliteration[char_norm]

                if not lower:
                    char_norm = char_norm.upper()
                if char != char_norm:
                    text = re.sub(re.escape(char), char_norm, text)
                    char = char_norm
                    char_lower = char.lower()

            '''Remove vowels, if required.'''
            if char not in letters_in and remove_vowels:
                if char_lower in vowels:
                    text = re.sub(re.escape(char), '', text)

            ''' Remove any character that is not in the alphabet, if otherwise specified'''
            if not copy_through and char not in letters_in and (char in letters_out or char_lower not in alphabet):
                text = re.sub(re.escape(char), '', text)


    '''Remove extra spaces'''
    text = re.sub('\s+', ' ', text).strip()

    return text

def is_punc(char):
    punct = (char != "'" and not re.match(digit, char) and (not char.isalnum() or bool(re.match(punctuation_symbol, char)) or bool(re.match(emoji_symbol, char))))
    return punct

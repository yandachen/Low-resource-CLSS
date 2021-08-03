NORMALIZATION V3.5 - 12/2020
Ramy Eskander
rnd2110@columbia.edu
=================================

Currently Supported Languages:
================================
- English
- Swahili
- Tagalog
- Somali
- Lithuanian
- Bulgarian
- Pashto
- Farsi
- Kazakh

Updates since V3.2
================================
- Added support for Kazakh
- Better coverage of punctuation marks, numbers and symbols.
- Numbers are checked before punctuation removal, so that numeric periods, for instance, remain intact.
- '&' is treated as a punctuation mark (as Python built-in punctuation checks are used)
- Isolated single quotations are differentiated from apostrophes.

Usage (default config):
===========================
import normalization
text = "some text"
normalized_text = normalization.process(language, text, letters_to_keep='', letters_to_remove='', lowercase=False, remove_repetitions_count=-1, remove_punct=False, remove_digits=False, remove_vowels=False, remove_diacritics=True, remove_spaces=False, remove_apostrophe=False, copy_through=True, keep_romanized_text=True)

Where the arguments are as follow:
1- language:string (case-insensitive): Material codes (e.g., 1A), ISO codes (e.g., SWA) and full language names (e.g., Swahili) are all accepted inputs.
2- text:string
3- letters_to_keep:string (case-sensitive): Letters needed to be kept, overwrites the removal of vowels, diacritics, non-alphabet characters and built-in language mappings -- "" means do not use this feature.
4- letters_to_remove:string (case-sensitive): Letters needed to be removed -- "" means do not use this feature.
5- lowercase:boolean
6- remove_repetitions_count:int: The maximum number of allowed character repetitions (in a sequence), e.g., when set to 2, “mannner” changes to “manner” -- 0 means do not use this feature (after the built-in mapping, lower-casing and removal of extras (e.g., non-zero width joiners) and before any other operations).
7- remove_punct:boolean: Covers both punctuation marks and symbols
8- remove_digits:boolean
9- remove_vowels:boolean: Does not cover the short-vowel diacritics in Pashto and Farsi, and does not affect non-alphabet characters of the underling languages
10- remove_diacritics:boolean
11- remove_spaces:boolean
12- remove_apostrophe:boolean
13- copy_through:boolean: When set to True, none of the foreign letters gets omitted.
14- keep_romanized_text:boolean: This argument works when the language has a non-Latin script (Bulgarian and Pashto). When set to True, none of the letters of the Romanized Bulgarian script (a-z + ŭĭui͡ei͡ai͡u), in the case of Bulgarian, and the Romanized Pashto script (a-z), in the case of Pashto) gets omitted. When set to False, the Romanized Bulgarian letters are transliterated into the Cyrillic script, in the case of Bulgarian, and the romanized Pashto letters are omitted (transliteration is Pashto is not supported).

Note: In the case of Pashto, text cleanup is always performed. This includes:
1- Converting some letters that are not in the original Pashto and Farsi scripts to their Pashto cognates
2- Normalizing the orthography by resolving the cases where two or more letters are used interchangeably (in an inconsistent manner)
3- (1) and (2) can be overwritten by using the letters_to_keep parameter.
4- In the case of Kazakh: 
 --- Arabic letters are always transliterated into Cyrillic. 
 --- Latin letters are transliterated when keep_romanized_text is set to False. 
 --- Latin homoglyphs are transliterated if the text has at least one Cyrillic character that belongs to the Kazakh script.
 --- Kazakh extended Cyrillic letters are not transliterated into regular Cyrillic.





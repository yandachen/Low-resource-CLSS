import re

digit = '[\d፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼零一二三四五六七八九十百千万億兆つ]'

number = '^\%?' + digit + '+(([\.\,\:\-\/\٫])?' + digit + ')*$'

punctuation_symbol = '[\_\”\`\.\.\,\;\:\?\!\[\]\{\}\(\)\|\~\<\>\ː\*\\\\/\+\-\=\%\$\#\@\^µˆ¹³¼º²Δ⁰¾ⅣⅢⅡ½ㆍφθ\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0656\u065d\u0670\u0618\u0619\u061A\u00ad]+'

emoji_symbol = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "]+)"
)

extras = "[\u200c\u0640\ufe0f]"

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
bulgarian_romanized_vowels = "aeiou" + bulgarian_vowels

pashto_diac = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0656\u065d\u0670\u0618\u0619\u061A"
pashto_alphabet = "ایردنهموتبسلشکزفگعخقيجحپصآطچضكظغذئثژأىءؤۀةھإ" + pashto_diac
pashto_romanized_alphabet = "abcdefghijklmnopqrstuvwxyz'" + pashto_alphabet
pashto_vowels = "اويےﮮۍیىې"
pashto_romanized_vowels = "aeiou" + pashto_vowels

farsi_diac = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0656\u065d\u0670\u0618\u0619\u061A"
farsi_alphabet = "ایردنهموتبسلشکزفگعخقيجحپصآطچضكظغذئثژأىءؤۀةھإ" + farsi_diac
farsi_romanized_alphabet = "abcdefghijklmnopqrstuvwxyz'" + farsi_alphabet
farsi_vowels = "اویۍۅۆېيىﯼﺎﺍﻮﻭﻰﻳﻴﻲﻱﯽﯾﯿےﮯ"
farsi_romanized_vowels = "aeiou" + farsi_vowels

kazakh_diac = "[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0656\u065d\u0670\u0618\u0619\u061A]+" #regex
kazakh_alphabet = "абвгдежзклмнопрстуфхчшыәғёиійқңөүұһцщъьэюя"
kazakh_romanized_alphabet = "abcdefƒghijklmnopqrstuvwxyzјàäćїúüóòöńñśşǵğýéè'" + kazakh_alphabet
kazakh_vowels = "әеіөүаыоұийуэ"
kazakh_romanized_vowels = "aeiouàäїúüóòöéè" + kazakh_vowels

georgian_alphabet = "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰჱჲჳჴჵ"
georgian_romanized_alphabet = "abcdeghijklmnopqrstuv'`" + georgian_alphabet
georgian_vowels = "აეიოუ"
georgian_romanized_vowels = "aeiou" + georgian_vowels
#diacritics: ა́ა̈ა̄ა̄̈ე́ე̄ი́ი̄ო́ო̈ო̄ო̄̈უ́უ̂უ̈უ̄უ̄̈ჷ́ჷ̄


alphabet_map = {}
alphabet_map["ENG"] = english_alphabet
alphabet_map["TGL"] = tagalog_alphabet
alphabet_map["SWA"] = swahili_alphabet
alphabet_map["SOM"] = somali_alphabet
alphabet_map["LIT"] = lithuanian_alphabet
alphabet_map["BUL"] = bulgarian_alphabet
alphabet_map["BUL_ROM"] = bulgarian_romanized_alphabet
alphabet_map["PUS"] = pashto_alphabet
alphabet_map["PUS_ROM"] = pashto_romanized_alphabet
alphabet_map["FAS"] = farsi_alphabet
alphabet_map["FAS_ROM"] = farsi_romanized_alphabet
alphabet_map["KAZ"] = kazakh_alphabet
alphabet_map["KAZ_ROM"] = kazakh_romanized_alphabet
alphabet_map["KAT"] = georgian_alphabet
alphabet_map["KAT_ROM"] = georgian_romanized_alphabet

vowels_map = {}
vowels_map["ENG"] = english_vowels
vowels_map["TGL"] = tagalog_vowels
vowels_map["SWA"] = swahili_vowels
vowels_map["SOM"] = somali_vowels
vowels_map["LIT"] = lithuanian_vowels
vowels_map["BUL"] = bulgarian_vowels
vowels_map["BUL_ROM"] = bulgarian_romanized_vowels
vowels_map["PUS"] = pashto_vowels
vowels_map["PUS_ROM"] = pashto_romanized_vowels
vowels_map["FAS"] = farsi_vowels
vowels_map["FAS_ROM"] = farsi_romanized_vowels
vowels_map["KAZ"] = kazakh_vowels
vowels_map["KAZ_ROM"] = kazakh_romanized_vowels
vowels_map["KAT"] = georgian_vowels
vowels_map["KAT_ROM"] = georgian_romanized_vowels

''' Special cases not handled by the default unicode undiacritization '''
diac_character_mappings = {
    'ą': 'a',
    'č': 'c',
    'ę': 'e',
    'ė': 'e',
    'į': 'i',
    'š': 's',
    'ų': 'u',
    'ū': 'u',
    'ž': 'z',
    'ø': 'o',
    'ӛ': 'ә',
    'ѓ': 'г',
    'ў': 'у'
}

# source: https://www.loc.gov/catdir/cpso/romanization/bulgarian.pdf
bulgarian_latin_transliteration = {
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

''' Special transformation for Pashto '''
pashto_character_mappings = {
    'ق': 'ک',
    'ف': 'پ',
    'ك': 'ک',
    'گ': 'ګ',
    'ﺉ': 'ي',
    'ئ': 'ي',
    'ہ': 'ه',
    'ھ': 'ه',
    'ٸ': 'ي',
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
    # numbers
    '٤': '۴',
    '٥': '۵',
    '٦': '۶',
    '0': '۰',
    '1': '۱',
    '2': '۲',
    '3': '۳',
    '4': '۴',
    '5': '۵',
    '6': '۶',
    '7': '۷',
    '8': '۸',
    '9': '۹'
}

farsi_character_mappings = {
    "آ": "ا",
    "أ": "ا",
    "إ": "ا",
    "ئ": "ی",
    "ى": "ی",
    "ي": "ی",
    "ؤ": "و",
    "ھ": "ه",
    "ۀ": "ه",
    'ة': 'ه',
    "ك": "ک",
    "ګ": "گ",
    "ڪ": "گ",
    "ټ": "ت",
    "ב": "پ",
    'ە': 'ه',
    'ې': 'ی',
    'ړ': 'ر',
    'ښ': 'س',
    'ہ': 'ه',
    'ٱ': 'ا',
    'ځ': 'خ',
    'ڵ': 'ل',
    'ٹ': 'ث',
    'څ': 'خ',
    'ڈ': 'د',
    'ډ': 'د',
    'ڕ': 'ر',
    'ۅ': 'و',
    'ڤ': 'ف',
    'ں': 'ن',
    'ڼ': 'ن',
    'ۆ': 'و',
    'ۍ': 'ی',
    # attachments
    'ﺁ': 'ا',
    'ﺆ': 'و',
    'ﺎ': 'ا',
    'ﺍ': 'ا',
    'ﺂ': 'ا',
    'ﺑ': 'ب',
    'ﺒ': 'ب',
    'ﺐ': 'ب',
    'ﺏ': 'ب',
    'ﭔ': 'ب',
    'ﭘ': 'پ',
    'ﺗ': 'ت',
    'ﺘ': 'ت',
    'ﺖ': 'ت',
    'ﺕ': 'ت',
    'ﺜ': 'ث',
    'ﺟ': 'ج',
    'ﺠ': 'ج',
    'ﺞ': 'ج',
    'ﺝ': 'ج',
    'ﭼ': 'چ',
    'ﭽ': 'چ',
    'ﺣ': 'ح',
    'ﺤ': 'ح',
    'ﺧ': 'خ',
    'ﺨ': 'خ',
    'ﺦ': 'خ',
    'ﺥ': 'خ',
    'ﺪ': 'د',
    'ﺩ': 'د',
    'ﺬ': 'ذ',
    'ﺫ': 'ذ',
    'ﺮ': 'ر',
    'ﺭ': 'ر',
    'ږ': 'ر',
    'ﺯ': 'ز',
    'ﺰ': 'ز',
    'ﮊ': 'ژ',
    'ﺳ': 'س',
    'ﺴ': 'س',
    'ﺲ': 'س',
    'ﺷ': 'ش',
    'ﺸ': 'ش',
    'ﺶ': 'ش',
    'ﺼ': 'ص',
    'ﺻ': 'ص',
    'ﺹ': 'ص',
    'ﻀ': 'ض',
    'ﻂ': 'ط',
    'ﻃ': 'ط',
    'ﻄ': 'ط',
    'ﻇ': 'ظ',
    'ﻆ': 'ظ',
    'ﻈ': 'ظ',
    'ﻋ': 'ع',
    'ﻌ': 'ع',
    'ﻊ': 'ع',
    '؏': 'ع',
    'ﻉ': 'ع',
    'ﻏ': 'غ',
    'ﻐ': 'غ',
    'ﻍ': 'غ',
    'ﻓ': 'ف',
    'ﻔ': 'ف',
    'ﻒ': 'ف',
    'ﻘ': 'ق',
    'ﻗ': 'ق',
    'ﻖ': 'ق',
    'ﻕ': 'ق',
    'ﻛ': 'ک',
    'ﻜ': 'ک',
    'ﻚ': 'ك',
    'ﮐ': 'ک',
    'ﮑ': 'ك',
    'ﮏ': 'ك',
    'ﮔ': 'گ',
    'ﮕ': 'گ',
    'ﮓ': 'گ',
    'ﮚ': 'گ',
    'ﮎ': 'ک',
    'ﻟ': 'ل',
    'ﻞ': 'ل',
    'ﻠ': 'ل',
    'ﻝ': 'ل',
    'ﻼ': 'لا',
    'ﻣ': 'م',
    'ﻤ': 'م',
    'ﻡ': 'م',
    'ﻢ': 'م',
    'ﻧ': 'ذ',
    'ﻨ': 'ذ',
    'ﻥ': 'ن',
    'ﻦ': 'ن',
    'ﻪ': 'ه',
    'ﻫ': 'ه',
    'ﻬ': 'ه',
    'ﻩ': 'ه',
    'ﮭ': 'ه',
    'ﮪ': 'ه',
    'ﮧ': 'ی',
    'ﻮ': 'و',
    'ﻭ': 'و',
    'ﻰ': 'ی',
    'ﻳ': 'ی',
    'ﻴ': 'ی',
    'ﻲ': 'ی',
    'ﻱ': 'ی',
    'ﯽ': 'ی',
    'ﯾ': 'ی',
    'ﯿ': 'ی',
    'ﯼ': 'ی',
    'ے': 'ی',
    'ﮯ': 'ی',
    'ﮮ': 'ی',
    'ێ': 'ی',
    'ۓ': 'ی',
    # numbers
    '٤': '۴',
    '٥': '۵',
    '٦': '۶',
    '0': '۰',
    '1': '۱',
    '2': '۲',
    '3': '۳',
    '4': '۴',
    '5': '۵',
    '6': '۶',
    '7': '۷',
    '8': '۸',
    '9': '۹'
}

farsi_latin_transliteration = {
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
    #    '‌': "=",
}

kazakh_character_mappings = {  # homoglyphs
    "A": "А",
    "a": "а",
    "B": "В",
    "C": "С",
    "c": "с",
    "E": "Е",
    "e": "е",
    "F": "Ғ",
    "H": "Н",
    "h": "һ",
    "I": "І",
    "i": "і",
    "K": "К",
    "k": "к",
    "M": "М",
    "O": "О",
    "o": "о",
    "P": "Р",
    "p": "р",
    "T": "Т",
    "X": "Х",
    "x": "х",
    "Y": "Ү",
    "y": "у",
    "Ë": "Ё",
    "ë": "ё",
    "Ӊ": "Ң",
    "ӊ": "ң",
    "Ə": "Ә",
    "ə": "ә"
}

kazakh_extended_cyrillic_transliteration = {
    "ә": "а",
    "ғ": "г",
    "ё": "е",
    "и": "ы",
    "і": "ы",
    "й": "ы",
    "қ": "к",
    "ң": "н",
    "ө": "о",
    "ү": "у",
    "ұ": "у",
    "һ": "х",
    "ц": "с",
    "щ": "ш",
    "ъ": "",
    "ь": ""
}

kazakh_latin_transliteration = {
    "ch": "ч",
    "ıa": "я",
    "ıo": "ё",
    "ıy": "и",
    "iy": "и",
    "kh": "х",
    "shch": "щ",
    "sh": "ш",
    "ya": "я",
    "yo": "ё",
    "yu": "ю",
    "yw": "ю",
    "zh": "ж",
    "a": "а",
    "b": "б",
    "c": "ш",  # с instead?
    "d": "д",
    "e": "е",
    "f": "ф",
    "ƒ": "ф",
    "g": "г",
    "h": "х",
    "i": "і",
    "j": "ж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "қ",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "ұ",
    "v": "в",
    "w": "у",  # ш instead?
    "x": "х",
    "y": "й",
    "z": "з",
    "ј": "ж",
    "ıý": "ю",
    "şç": "щ",
    "à": "ә",
    "ä": "ә",
    "ć": "ч",
    "é": "э",
    "è": "е",
    "ǵ": "ғ",
    "ğ": "ғ",
    "ń": "ң",
    "ñ": "ң",
    "ó": "ө",
    "ò": "ө",
    "ö": "ө",
    "ś": "ш",
    "ş": "ш",
    "ú": "ү",
    "ü": "ү",
    "ý": "у",
    "ї": "и"
}

kazakh_arabic_transliteration = {
    "تس": "ц",
    "شش": "щ",
    "يا": "я",
    "يۋ": "ю",
    "ء": "ә",
    "ﺃ": "ә",
    "أ": "ә",
    "ﺄ": "ә",
    "ﺆ": "о",
    "ﺋ": "ы",
    "ﺍ": "а",
    "ﺎ": "а",
    "ا": "а",
    "ٵ": "а",
    "ب": "б",
    "ﺑ": "б",
    "ﺒ": "б",
    "ﺏ": "б",
    "پ": "п",
    "ت": "т",
    "ﺘ": "т",
    "ﺗ": "т",
    "ﺕ": "т",
    "ﺖ": "т",
    "ج": "ж",
    "ﺟ": "ж",
    "ﺠ": "ж",
    "ﺞ": "ж",
    "چ": "ч",
    "ح": "х",
    "ﺤ": "х",
    "ﺣ": "х",
    "ﺡ": "х",
    "د": "д",
    "ﺪ": "д",
    "ﺩ": "д",
    "ر": "р",
    "ﺮ": "р",
    "ﺭ": "р",
    "ز": "з",
    "ﺰ": "з",
    "س": "с",
    "ﺴ": "с",
    "ﺳ": "с",
    "ﺲ": "с",
    "ﺱ": "с",
    "ش": "ш",
    "ﺸ": "ш",
    "ﺷ": "ш",
    "ﺵ": "ш",
    "ع": "г",
    "ﻋ": "г",
    "ﻌ": "г",
    "ﻊ": "г",
    "ﻉ": "г",
    "ﻐ": "ғ",
    "ﻏ": "ғ",
    "غ": "ғ",
    "ف": "ф",
    "ﻓ": "ф",
    "ﻔ": "ф",
    "ﻑ": "ф",
    "ﻒ": "ф",
    "ق": "қ",
    "ﻘ": "қ",
    "ﻗ": "қ",
    "ﻕ": "қ",
    "ك": "к",
    "ﻜ": "к",
    "ﻛ": "к",
    "ﻙ": "к",
    "ک": "к",
    "ڭ": "ң",
    "گ": "г",
    "ل": "л",
    "ﻟ": "л",
    "ﻠ": "л",
    "ﻝ": "л",
    "ﻞ": "л",
    "ﻷ": "ла",
    "ﻻ": "ла",
    "ﻼ": "ла",
    "م": "м",
    "ﻣ": "м",
    "ﻤ": "м",
    "ﻡ": "м",
    "ﻢ": "м",
    "ن": "н",
    "ﻨ": "н",
    "ﻦ": "н",
    "ﻥ": "н",
    "ه": "һ",
    "ﻬ": "һ",
    "ﻫ": "һ",
    "ﻪ": "һ",
    "ﻩ": "һ",
    "ھ": "һ",
    "ە": "һ",
    "و": "о",
    "ﻮ": "о",
    "ﻭ": "о",
    "ٶ": "ө",
    "ۆ": "в",
    "ۇ": "ұ",
    "ۋ": "о",
    "ى": "ы",
    "ﻰ": "ы",
    "ﻯ": "ы",
    "ی": "ы",
    "ي": "й",
    "ﻴ": "й",
    "ﻲ": "й",
    "ﻳ": "й",
    "ﻱ": "й",
    "ﺌ": "и",
    "ﻧ": "н",
}

georgian_latin_transliteration = {
    "a": "ა",
    "b": "ბ",
    "ch'ʼ": "ჭ",
    "chʼ": "ჭ",
    "ch": "ჩ",
    "dz": "ძ",
    "d": "დ",
    #"ej": "ჱ",
    "e": "ე",
    "gh": "ღ",
    "g": "გ",
    "h": "ჰ",
    "i": "ი",
    #"j": "ჲ",
    "j": "ჯ",
    "kh": "ხ",
    "k'ʼ": "კ",
    "kʼ": "კ",
    "k": "ქ",
    "l": "ლ",
    "m": "მ",
    "n": "ნ",
    "o": "ო",
    #"ȯ": "ჵ",`
    "p'": "პ",
    "pʼ": "პ",
    "p": "ფ",
    "q'": "ყ",
    "qʼ": "ყ",
    #"q̌": "ჴ",
    "r": "რ",
    "sh": "შ",
    "s": "ს",
    "ts'": "წ",
    "tsʼ": "წ",
    "ts": "ც",
    "t'": "ტ",
    "tʼ": "ტ",
    "t": "თ",
    "u": "უ",
    #"ŭ": "ჳ",
    "v": "ვ",
    "zh": "ჟ",
    "z": "ზ",
}
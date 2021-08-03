# coding=utf-8

import os, sys

current_path = os.path.abspath('.')

print(current_path)

parent_path = os.path.dirname(current_path)

sys.path.append(parent_path)

import normalization

text = "tai mažas testas, taip vaikinaiūy ĖŠ aąbcčdeęėfghiįyjklmnoprsštuųūvzž"
normalized_text = normalization.process('LT', text, letters_to_keep='', letters_to_remove='', lowercase=False, remove_repetitions_count=-1, remove_punct=False, remove_digits=False, remove_vowels=False, remove_diacritics=True, remove_spaces=False, remove_apostrophe=False, copy_through= True, keep_romanized_text=True)
print(normalized_text)

text = "Côte d'Ivoire ۶ 汉字 I need to keep these line ĖŠ ﺭژیﻡ ﺺﻫیﻮﻧیﺲﺗی ﻭ ﻊﻣﺎﻧ ﻢﻧﺎﺴﺑﺎﺗ ﺩیپﻞﻣﺎﺗیک ﻥﺩﺍﺮﻧﺩ  The Zionist regime and Oman do not have diplomatic relations ﻢﻧ ﻢﻧ پﻩ ﻝښ ﻖﻗ٤ﻖﻔﻘﻘﻛﺭگﺎﻫ ﻙې ﺩ پﻮﻠﻴﻔﻔﺳﻭﺩ ﻝﻮﻣړﻯ ﺎﻤﻧ ﺲﻴﺘﻳ  abcﺡﻭﺯې ﻞﻫ   ﺁﺁ ﺂﻣﺮﺳﺮﻫ ﻡﺮﻜﻫ ٪ ﻢﻧ. ؟. ﻢﻧ ،  ﻢَﻧ ﻢٍﻧ ﻝ  ١٢٣٤٥٦ ۴ ۴ ﻢﻤﻤﻤﻤﻤﻤﻣ '''This is some test "
normalized_text = normalization.process("FA", text, copy_through=True, remove_diacritics=True, keep_romanized_text=False)
print(normalized_text.encode("utf-8"))
print(normalized_text)

text = "Côte d'Ivoire ۶ 汉字 I need to keep these line ĖŠ من من په لښ قق٤قفققكرگاه كې د پوليففسود لومړى امن سيتي  abcحوزې له   آآ آمرسره مركه ٪ من. ؟. من ،  مَن مٍن ل  ١٢٣٤٥٦ ۴ ۴ مممممممم '''This is some test "
normalized_text = normalization.process("PS", text, copy_through=False, remove_diacritics=True, keep_romanized_text=False)
print(normalized_text.encode("utf-8"))
#print(normalized_text)

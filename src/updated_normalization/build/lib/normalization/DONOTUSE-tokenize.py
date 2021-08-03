# -*- coding: UTF-8 -*-

import re
import sys

def clean(content):
    # 1) replace ZWNJ by space
    content = content.replace('\u200c', ' ')
    # 2) replace hyphen by space
    # - is a hyphen-minus (ASCII 2D, Unicode 002D), normally used as a hyphen,
    # or in math expressions as a minus sign
    # – is an en dash (Unicode 2013). This can also be entered from the Special
    # characters: Symbols bar above the text-entry field; it's between the m³
    # and —
    # — is an em dash (Unicode 2014). This can also be entered from the Special
    # characters: Symbols bar; it's between the – and …
    # − is a minus (Unicode 2212). This can also be entered from the Special
    # characters: Symbols bar; it's between the ± and ×
    hyphens_list = ['\u1806', '\u002d', '\u2010', '\u2011', '\u2012', '\u2013',
        '\u2014', '\u2015', '\u2212']
    hyphens_regex = '[' + ''.join(hyphens_list) + ']'
    re.sub(hyphens_regex, ' ', content)
    # 3) split by space.
    return content.split()

def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as fp:
        for line in fp:
            print(clean(line))

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
    you can get the source and tessdata from https://github.com/tesseract-ocr
    before use pyocr, you should install tesseract(https://github.com/tesseract-ocr/tesseract/wiki)
    if you want to add tessdata, you can download tessdata file and add it to /usr/share/tesseract-ocr/tessdata
"""
import sys
from PIL import Image
import pyocr
import pyocr.builders


if __name__ == "__main__":
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    print tools
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    print("Will use tool '%s'" % (tool.get_name()))
    # Ex: Will use tool 'libtesseract'

    langs = tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    lang = langs[0]
    print("Will use lang '%s'" % (lang))
    # Ex: Will use lang 'fra'
    # Note that languages are NOT sorted in any way. Please refer
    # to the system locale settings for the default language
    # to use.

    txt = tool.image_to_string(
        Image.open('../data/1.png'),
        lang="chi_sim",
        builder=pyocr.builders.TextBuilder()
    )
    print txt
    print "微信" in txt.encode('utf-8')


import re
from xml.dom import minidom
import pandas as pd
import xml
from translate import Translator


def TS_2_CSV(xml_filepath,csv_savepath):
    DOMTree = xml.dom.minidom.parse(xml_filepath)
    collection = DOMTree.documentElement

    sources = collection.getElementsByTagName("source")
    translations = collection.getElementsByTagName("translation")
    chinese = []
    english = []

    for source, translation in zip(sources, translations):
        if source.childNodes:
            data = source.childNodes[0].data
        else:
            data = 'none'
        chinese.append(data)
        data = 'unfinished: '
        if translation.childNodes:
            if translation.hasAttribute('unfinished'):
                data = data + translation.childNodes[0].data
            else:
                data = translation.childNodes[0].data
        english.append(data)

    pd_dict = {'chinese': chinese, 'english': english}
    stat = pd.DataFrame(pd_dict)
    stat.to_csv(csv_savepath, encoding='utf-8-sig')


def is_Chinese(word):
    for ch in word:
        if '\u4e00' > ch or ch > '\u9fff':
            return False
    return True

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')  # 匹配汉字的正则表达式
    return bool(pattern.search(text))


def translate_ts(filepath,outpath):
    DOMTree = xml.dom.minidom.parse(filepath)
    collection = DOMTree.documentElement
    translator = Translator(from_lang="zh", to_lang="en")
    sources = collection.getElementsByTagName("source")
    translations = collection.getElementsByTagName("translation")
    # c=0
    for source_element, translation_element in zip(sources, translations):
        # if c==50:
        #     break
        source_child = source_element.firstChild
        if source_child and source_child.nodeValue:
            if contains_chinese(source_child.nodeValue):
                translated_text = translator.translate(source_child.nodeValue)
            else:
                translated_text=''
        # 创建新的文本节点并设置翻译结果
            if translated_text and not translation_element.firstChild:
                new_translation_text = DOMTree.createTextNode(translated_text)
                translation_element.appendChild(new_translation_text)
                translation_element.removeAttribute("type")
                # print(t_child)
                # .nodeValue = translated_text
        # c+=1
    # 保存修改后的 XML 数据回文件
    with open(outpath, "w",encoding='utf-8') as file:
        file.write(DOMTree.toxml())



if __name__ == '__main__':
    translate_ts('translate/dentalnavi_en.ts','translate/dentalnavi_en1.ts')
    # TS_2_CSV('spdImplant_en1.ts','spdImplant_en1.csv')


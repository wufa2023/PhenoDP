from bs4 import BeautifulSoup
import re
import json

import re

def remove_references(content):
    # 使用正则表达式删除引用内容（例如：xxx et al.）及其后面的一个字符
    content = re.sub(r'\b\w+\s+et\s+al\.\b.', '', content)
    return content

def remove_parentheses(content):
    # 使用正则表达式删除括号及其内部的内容及其后面的一个字符
    return re.sub(r'\([^()]*\).', '', content)

def remove_last(content):
    # 删除包含指定字符串的内容及其后面的一个字符
    content = re.sub(r'\bsee\s+\w+\b.', '', content)
    content = re.sub(r'see\d+.', '', content)
    return content


with open("../source_data/omim_to_hpo.json", "r") as f:
    omim_data = json.load(f)

omim_context_map = {}

completed_omim_numbers = list(set(omim_data.keys()))
ids = completed_omim_numbers


# 读取HTML文件内容
for i in range(len(ids)):

    omim_id = ids[i].split(':')[-1]
    print(omim_id, i, '/', len(ids))
    with open("../source_data/entry/{}.html".format(omim_id), "r", encoding="utf-8") as f:
        html_content = f.read()

    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 找到指定ID的标签
    target_tag = soup.find(id="mimDescriptionFold")

    # 提取目标标签内的内容并删除括号及其内部内容
    if target_tag:
        content = target_tag.get_text(strip=True)
        print(content)
        content_without_parentheses = remove_parentheses(content)
        context = remove_references(content_without_parentheses)
        context_1 = context.rsplit(',', 1)[:-1]
        context_2 = context.rsplit(',', 1)[-1]
        context_2 = remove_last(context_2)
        context = ''.join(context_1)
        context = context +','+ ''.join(context_2)

        while context[-1] in [',', '.', ' ']:
            context = context[:-1]
        context = context + '.'
        context = re.sub(r'\s+', ' ', context)
        if context[0] == ',':
            context = context[1:]
        print(context)
        omim_context_map[omim_id] = context
        print('')
with open("../source_data/omim_context.json", "w") as outfile:
    json.dump(omim_context_map, outfile, indent=4)
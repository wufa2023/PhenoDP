from bs4 import BeautifulSoup
import re
import json

def extract_title_and_abbreviation_from_entry(entry):
    # 使用正则表达式提取第二个-之后的全部内容，包括-
    match = re.search(r'(?<=- ).*', entry)
    if match:
        return match.group(0)
    else:
        return None

def extract_titles_from_html(html_content):
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 找到所有的title标签
    title_tags = soup.find_all('title')

    # 提取title标签的内容
    titles = [tag.get_text() for tag in title_tags]

    return titles

# 读取OMIM到HPO的映射数据
with open("../source_data/omim_to_hpo.json", "r") as f:
    omim_data = json.load(f)

completed_omim_numbers = list(set(omim_data.keys()))
ids = completed_omim_numbers

omim_disease_names = {}

for i in range(len(ids)):
    print(ids[i], i, '/ ', len(ids))
    omim_id = ids[i].split(':')[-1]
    with open('../source_data/entry/{}.html'.format(omim_id), 'r', encoding='utf-8') as file:
        html_content = file.read()

    titles = extract_titles_from_html(html_content)

    if titles:
        title = titles[0]
        disease_name = extract_title_and_abbreviation_from_entry(title).split(' - ')[-1]
        omim_disease_names[omim_id] = disease_name
        print('   ', disease_name)
# 将结果保存到JSON文件中
with open("../source_data/omim_disease_names.json", "w") as outfile:
    json.dump(omim_disease_names, outfile)

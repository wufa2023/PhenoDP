import pandas as pd
import requests
import time
import os
import json

def spider(url, omim_number, success_omim_file, failed_omim_file):
    headers = {
        'User-Agent': 'bingbot (+https://www.bing.com/bingbot.htm)'
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            html_content = response.text
            with open(f"../source_data/entry_html/{omim_number}.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"Page content saved to entry_html/{omim_number}.html")
            success_omim_file.write(f"{omim_number}\n")
            success_omim_file.flush()  # 立即刷新文件缓冲区，确保写入文件
            time.sleep(4.5)
            return True
        else:
            print(f"Request failed with status code {response.status_code}, OMIM number: {omim_number}")
            failed_omim_file.write(f"{omim_number}\n")
            failed_omim_file.flush()  # 立即刷新文件缓冲区，确保写入文件
            return False
    except requests.RequestException as e:
        print(f"Request exception: {e}, OMIM number: {omim_number}")
        failed_omim_file.write(f"{omim_number}\n")
        failed_omim_file.flush()  # 立即刷新文件缓冲区，确保写入文件
        return False


# 读取OMIM JSON文件
with open("../source_data/omim_to_hpo.json", "r") as f:
    omim_data = json.load(f)

# 提取OMIM编号
completed_omim_numbers = set(omim_data.keys())

# 打开文件准备写入成功和失败的OMIM编号
with open("../source_data/success_omim_numbers.txt", "a") as success_omim_file, open("../source_data/failed_omim_numbers.txt", "a") as failed_omim_file:
    for omim_number in completed_omim_numbers:
            url = f"https://www.omim.org/entry/{omim_number.split(':')[-1]}"
            success = spider(url, omim_number, success_omim_file, failed_omim_file)
            if success:
                completed_omim_numbers.add(omim_number)


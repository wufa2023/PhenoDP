import json
import random
import re

# 读取 JSON 文件
with open("../source_data/omim_context.json", "r") as infile:
    omim_context_map = json.load(infile)

# 定义要删除的短语列表
phrases_to_remove = ["is an", "is a", "is characterized", "is the", "is defined", "is diagnosed", 'represents',
                     "comprises", "is considered", "is caused", "are a group of",
                     "are here classified", "refers to", "is due to", "presents with", "is one of",
                     "leads to", "encompasses", "leads to", "include", "is usually diagnosed", "are",
                     "consists of", "is", "refer to", "results from", "results in"]
matched_omim_context = {}

with open("../source_data/missMatch.txt", "w") as outfile:
    count = 0
    for omim_id, context in list(omim_context_map.items()):
        count += 1
        print(count, '/', len(list(omim_context_map)))
        state = 0
        for phrase in phrases_to_remove:
            match = re.search(rf'\b{phrase}\b', context)
            if match:
                state = 1
                start_index = match.start()
                context = phrase + ' ' + context[start_index + len(phrase):].lstrip(" .")

                break
        if state == 0:
            outfile.write(f"{omim_id}\n")
        matched_omim_context[omim_id] = context

with open("../source_data/omim_context_match.json", "w") as outfile:
    json.dump(matched_omim_context, outfile, indent=4)
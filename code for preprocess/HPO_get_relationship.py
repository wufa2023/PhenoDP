import time
import requests
import pandas as pd

# 读取包含多条 HPO 号的 CSV 文件
hpo_df = pd.read_csv("../source_data/hpo_numbers.csv")
hpo_df = hpo_df.iloc[1:, ]
hpo_numbers = hpo_df["HPO Number"]

# 打开文件准备写入结果、失败的 HPO 号和异常信息
with open("../source_data/hpo_relations.txt", "w") as hpo_relations_file, \
        open("../source_data/log_relationship_failed.txt", "w") as failed_hpos_file, \
        open("../source_data/log_relationship_error.txt", "w") as error_log_file:
    # 遍历每个 HPO 号
    count = 0
    for hpo_number in hpo_numbers:
        count += 1
        print(count, '/', len(hpo_numbers), hpo_number)
        # 构建 URL
        url = "https://hpo.jax.org/api/hpo/term/" + str(hpo_number)

        try:
            # 获取数据
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()

                # 提取父子关系信息
                details = data.get('details', {})
                relations = data.get('relations', {})
                parents = relations.get('parents', [])
                children = relations.get('children', [])

                # 将结果写入文件
                hpo_relations_file.write(f"term-{hpo_number}\n")
                hpo_relations_file.write("Parents:\n")
                for parent in parents:
                    hpo_relations_file.write(f"    {parent['ontologyId']} - {parent['name']}\n")

                hpo_relations_file.write("Children:\n")
                for child in children:
                    hpo_relations_file.write(f"    {child['ontologyId']} - {child['name']}\n")

            else:
                print(f"Failed to retrieve data for HPO number {hpo_number}. Status code:", response.status_code)
                failed_hpos_file.write(f"{hpo_number}\n")
        except Exception as e:
            print(f"An error occurred while processing HPO number {hpo_number}: {e}")
            error_log_file.write(f"{hpo_number}: {e}\n")
            continue  # 继续循环
        time.sleep(0.5)

print("HPO relations saved to hpo_relations.txt")
print("Failed HPO numbers saved to log_relationship_failed.txt")
print("Error log saved to log_relationship_error.txt")

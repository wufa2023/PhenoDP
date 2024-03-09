import time
import requests
import pandas as pd

# 读取包含多条 HPO 号的 CSV 文件
hpo_df = pd.read_csv("../source_data/hpo_numbers.csv")
hpo_df = hpo_df.iloc[1:, ]
hpo_numbers = hpo_df["HPO Number"]

# 打开文件准备写入结果、失败的 HPO 号和异常信息
with open("../source_data/hpo_to_omim_results.txt", "w") as omim_file, \
        open("../source_data/log_hpo_failed.txt", "w") as failed_hpos_file, \
        open("../source_data/log_hpo_error.txt", "w") as error_log_file:
    # 遍历每个 HPO 号
    count = 0
    for hpo_number in hpo_numbers:
        count += 1
        print(count, '/', len(hpo_numbers), hpo_number)
        # 构建 URL
        url = "https://hpo.jax.org/api/hpo/term/" + str(hpo_number) + "/diseases"

        try:
            # 获取数据
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['diseases'])

                omim_df = df.loc[df['db'] == 'OMIM']

                omim_numbers = omim_df['diseaseId'].tolist()

                # 将结果写入文件
                for omim_number in omim_numbers:
                    omim_file.write(f"{hpo_number},{omim_number}\n")
            else:
                print(f"Failed to retrieve data for HPO number {hpo_number}. Status code:", response.status_code)
                failed_hpos_file.write(f"{hpo_number}\n")
        except Exception as e:
            print(f"An error occurred while processing HPO number {hpo_number}: {e}")
            error_log_file.write(f"{hpo_number}: {e}\n")
            continue  # 继续循环


print("OMIM numbers saved to omim_results.txt")
print("Failed HPO numbers saved to failed_hpos.txt")
print("Error log saved to error_log.txt")

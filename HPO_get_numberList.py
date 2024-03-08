import csv

hpo_numbers = []

with open("hp.obo", "r", encoding="utf-8") as file:
    lines = file.readlines()

    for line in lines:
        if line.startswith("id: HP:"):
            hpo_number = line.strip().split(" ")[1]
            hpo_numbers.append(hpo_number)

# 将 HPO 号保存为 CSV 文件
with open("hpo_numbers.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["HPO Number"])

    for hpo_number in hpo_numbers:
        writer.writerow([hpo_number])

print("HPO Numbers saved to hpo_numbers.csv")

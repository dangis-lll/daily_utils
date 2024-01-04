import csv

input_file = r'files/SCAResult.txt'
output_file = "files/output.csv"

# 打开输入文件和输出CSV文件
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline="") as csvfile:
    # 创建CSV写入器
    csv_writer = csv.writer(csvfile)

    # 写入表头
    csv_writer.writerow(["代码位置", "警告类型", "详细"])

    lines = []
    for line in infile:
        # 提取信息
        if line.startswith(
                "C:\\C++_program\\xindaima\\release-build") and 'out\\build' not in line and "thirdparty" not in line:
            line = line.replace("C:\\C++_program\\xindaima\\release-build\\", "")
            parts = line.split(": ")
            code_location = parts[0]
            warning_type = parts[1]
            detail = "".join(parts[2:]).strip()  # 移除前后的空白字符
            lines.append([code_location, warning_type, detail])

    # 根据首字母对行进行排序
    sorted_lines = sorted(lines, key=lambda x: x[0])

    # 将排序后的行写入CSV文件
    for line in sorted_lines:
        csv_writer.writerow(line)
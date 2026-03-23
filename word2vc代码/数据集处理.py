# 创建时间    : 2026/3/13 08:27
# 作者       : 叶之瞳
# 文件名      : 数据集处理.py
# 读取原始文件
with open('Test1.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 计算要保留的行数（一半）
half_count = len(lines) // 2

# 将前一半写入新文件
with open('Test.txt', 'w', encoding='utf-8') as f:
    f.writelines(lines[:half_count])

print(f"原始数据共 {len(lines)} 行，已保留前 {half_count} 行到 Test.txt")
# 导入所需库
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- 1. 读取Excel文件 ----------------------
# 请确保Excel文件（depth_compare.xlsx）与代码在同一目录下
# 若不在同一目录，请将'depth_compare.xlsx'替换为文件的完整路径（如：'C:/Users/xxx/Desktop/depth_compare.xlsx'）
file_path = 'depth_compare.xlsx'
try:
    df = pd.read_excel(file_path, sheet_name='Sheet1')  # 读取Sheet1
    print("Excel文件读取成功！")
    print("原始数据预览：")
    print(df.head())  # 打印前几行，确认数据正确
except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}，请检查文件路径是否正确。")
    exit()  # 退出程序

# ---------------------- 2. 计算差值 ----------------------
# 确保Excel中存在"真实值"和"实验值"列
if '真实值' not in df.columns or '实验值' not in df.columns:
    print("错误：Excel中未找到'真实值'或'实验值'列，请检查列名。")
    exit()

df['差值'] = df['实验值'] - df['真实值']  # 实验值-真实值的差值
df['差值绝对值'] = df['差值'].abs()     # 差值的绝对值
# 打印计算结果
print("\n真实值、实验值及差值计算结果：")
print(df.round(4))  # 保留4位小数

# ---------------------- 3. 绘制矢量图 ----------------------
# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示（Windows用SimHei，MAC用'Arial Unicode MS'）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.figure(figsize=(10, 6), dpi=100)         # 画布大小

# 绘制折线+散点
plt.plot(df['真实值'], df['差值绝对值'], color='#1f77b4', linewidth=2, label='Absolute Difference')
plt.scatter(df['真实值'], df['差值绝对值'], color='#1f77b4', s=60, zorder=5)

# 为每个点标注具体数值
for x, y in zip(df['真实值'], df['差值绝对值']):
    plt.text(x, y + 0.01, f'{y:.2f}', ha='center', va='bottom', fontsize=9)

# 设置坐标轴和标题
plt.xlabel('True Value(cm)', fontsize=12, fontweight='bold')
plt.ylabel('Absolute Difference(cm)', fontsize=12, fontweight='bold')
plt.title('Distribution of Absolute Differences between True and Experimental Values', fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='-')
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()

# ---------------------- 4. 保存矢量图+显示图表 ----------------------
# 保存为SVG矢量格式（也可改为'pdf'/'eps'）
plt.savefig('真实值-实验值差值绝对值矢量图.svg', format='svg', bbox_inches='tight')
print("\n矢量图已保存为：真实值-实验值差值绝对值矢量图.svg")
plt.show()

# 打印统计信息
print("\n差值绝对值统计信息：")
print(f"平均值：{df['差值绝对值'].mean():.4f}")
print(f"最大值：{df['差值绝对值'].max():.4f}")
print(f"最小值：{df['差值绝对值'].min():.4f}")
print(f"标准差：{df['差值绝对值'].std():.4f}")
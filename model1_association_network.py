import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 读取数据, 按订单分组生成SKU集合
time1 = time.time()
# 读取Excel文件并筛选6月份数据
df = pd.read_excel(r"D:\Mycode\KIVA\data\outbound7-23.xlsx")
df['date'] = pd.to_datetime(df['date_crt'])  # 将日期列转换为datetime格式
# df = df[df['date'].dt.month == 6]  # 只保留6月份的数据
df['mat_code'] = df['mat_code'].astype(str)  # 关键修复步骤！
time2 = time.time()
print(f"读取数据成功！用时{time2-time1:.2f}s, 数据大小:", df.shape)
# print(df.head(5))

# 按订单分组生成SKU集合
# orders = df.groupby(by='order_num') # 按订单编号分组,生成一个DataFrameGroupBy对象, 分组后对象不是DataFrame，而是中间状态对象
# orders = orders['mat_code'].apply(set) # 输出：一个Series，索引为订单号，值为该订单包含的唯一sku集合（如：{A001, B005}），并转换为集合（set），以去除重复的sku编码。
orders = df.groupby(by='order_num')['mat_code'].apply(set) # 按订单编号分组，并将每个订单中的sku编码转换为集合（set），以去除重复的sku编码。
orders = orders.reset_index(name='skus') # 将订单号从索引恢复为数据列,将结果重置为一个新的 DataFrame，并将sku编码集合命名为 skus 列。
print(orders)

# # 保存分组后的数据集
# orders.to_excel(r"D:\Mycode\KIVA\result\problem1\orders.xlsx", index=False)
transaction_data = [list(order) for order in orders['skus']]

# 转换为Apriori算法所需的二进制矩阵
te = TransactionEncoder()
te_data = te.fit(transaction_data).transform(transaction_data)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)
print(df_encoded)

# 保存编码后的数据集
# df_encoded.to_excel(r"D:\Mycode\KIVA\result\problem1\encoded_data.xlsx", index=False)

# 2. 构建商品关联矩阵（基于共现频次）
print("\n开始构建商品关联矩阵...")
time3 = time.time()

# 获取所有商品列表
skus = list(te.columns_)
num_skus = len(skus)
print(f"商品总数: {num_skus}")

# 初始化关联矩阵
association_matrix = np.zeros((num_skus, num_skus))

# 计算商品间的共现频次
for i in range(num_skus):
    for j in range(i, num_skus):  # 只计算上三角矩阵，因为关联是对称的
        if i == j:
            # 对角线元素：商品自身出现的频次
            association_matrix[i][j] = df_encoded.iloc[:, i].sum()
        else:
            # 非对角线元素：两个商品共同出现的频次
            co_occurrence = (df_encoded.iloc[:, i] & df_encoded.iloc[:, j]).sum()
            association_matrix[i][j] = co_occurrence
            association_matrix[j][i] = co_occurrence  # 对称矩阵

# 将关联矩阵转换为DataFrame便于查看和保存
association_df = pd.DataFrame(association_matrix, index=skus, columns=skus)
print(f"关联矩阵构建完成！用时{time.time()-time3:.2f}s")
print(f"关联矩阵形状: {association_df.shape}")

# 保存关联矩阵
# association_df.to_excel(r"D:\Mycode\KIVA\result\problem1\association_matrix.xlsx")
# print("关联矩阵已保存到: D:\Mycode\KIVA\result\problem1\association_matrix.xlsx")

# 3. 构建和可视化商品关联网络
print("\n开始构建商品关联网络...")
time4 = time.time()

# 设置关联阈值（只显示共现频次大于阈值的关联）
threshold = 10  # 可以根据数据调整
print(f"关联阈值设置为: {threshold}")

# 创建网络图
G = nx.Graph()

# 添加节点（商品）
for sku in skus:
    G.add_node(sku)

# 添加边（商品关联）
edge_count = 0
for i in range(num_skus):
    for j in range(i+1, num_skus):  # 避免重复添加边
        weight = association_matrix[i][j]
        if weight >= threshold:
            G.add_edge(skus[i], skus[j], weight=weight)
            edge_count += 1

print(f"网络节点数: {G.number_of_nodes()}")
print(f"网络边数: {G.number_of_edges()}")
print(f"关联网络构建完成！用时{time.time()-time4:.2f}s")

# 4. 绘制关联网络图
print("\n开始绘制关联网络图...")
time5 = time.time()

plt.figure(figsize=(15, 12))

# 使用布局算法（兼容旧版本scipy）
try:
    pos = nx.spring_layout(G, k=3, iterations=50)
except AttributeError:
    # 如果spring_layout出现兼容性问题，使用其他布局
    if G.number_of_nodes() > 0:
        pos = nx.circular_layout(G)  # 圆形布局作为备选
    else:
        pos = {}

# 绘制节点
nx.draw_networkx_nodes(G, pos, 
                      node_color='lightblue', 
                      node_size=500, 
                      alpha=0.8)

# 绘制边，边的粗细根据权重调整
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights) if weights else 1
normalized_weights = [w/max_weight * 5 for w in weights]  # 归一化权重用于边的粗细

nx.draw_networkx_edges(G, pos, 
                      width=normalized_weights,
                      alpha=0.6,
                      edge_color='gray')

# 绘制节点标签
nx.draw_networkx_labels(G, pos, 
                       font_size=8, 
                       font_weight='bold')

# 添加边权重标签（可选，数据量大时可能会很拥挤）
if G.number_of_edges() <= 50:  # 只在边数较少时显示权重
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

plt.title(f'商品关联网络图\n(关联阈值: {threshold}, 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()})', 
         fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()

# 保存网络图
plt.savefig(r"D:\Mycode\KIVA\result\association_network.png", dpi=300, bbox_inches='tight')
print(f"关联网络图绘制完成！用时{time.time()-time5:.2f}s")
print("网络图已保存到: D:\Mycode\KIVA\result\problem1\association_network.png")

# 显示图形
plt.show()

# 5. 输出网络统计信息
print("\n=== 网络统计信息 ===")
print(f"总处理时间: {time.time()-time1:.2f}s")
print(f"商品总数: {num_skus}")
print(f"订单总数: {len(orders)}")
print(f"网络节点数: {G.number_of_nodes()}")
print(f"网络边数: {G.number_of_edges()}")
print(f"网络密度: {nx.density(G):.4f}")

# 输出度最高的前10个商品
if G.number_of_nodes() > 0:
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n度最高的前10个商品:")
    for node, degree in top_nodes:
        print(f"  {node}: {degree}")

print("\n商品关联网络分析完成！")

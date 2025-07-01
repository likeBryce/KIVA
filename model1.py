import pandas as pd
import time 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. 读取数据, 按订单分组生成SKU集合
time1 = time.time()
# 读取Excel文件并筛选6月份数据
df = pd.read_excel(r"D:\Mycode\KIVA\data\outbound.xlsx")
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
# print(orders)
# 保存分组后的数据集
orders.to_excel(r"D:\Mycode\KIVA\result\orders.xlsx", index=False)
transaction_data = [list(order) for order in orders['skus']]

# 转换为Apriori算法所需的二进制矩阵
te = TransactionEncoder()
te_data = te.fit(transaction_data).transform(transaction_data)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)
# print(df_encoded)
# 保存编码后的数据集
# df_encoded.to_excel(r"D:\Mycode\KIVA\result\encoded_data.xlsx", index=False)

# 2.使用 Apriori 挖掘频繁项集
# 设置最小支持度（假设为2%）
min_support = 0.02
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# 按项集大小和频率排序
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets.sort_values(by=['length', 'support'], ascending=False)
frequent_itemsets.to_excel(r"D:\Mycode\KIVA\result\frequent_itemsets.xlsx", index=False)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("\n关联规则:")
rules.to_excel(r"D:\Mycode\KIVA\result\association_rules.xlsx", index=False)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]) #前项、后项、支持度、置信度和提升度

# # 3. 货架分配策略
# racks = [] # 存储所有货架，每个货架是一个SKU列表
# used_skus = set() # 记录已经被分配到货架的SKU

# for _, row in frequent_itemsets.iterrows(): # 遍历所有频繁项集（即经常一起购买的商品组合）
#     items = list(row['itemsets']) # 获取当前项集的SKU列表
#     if len(items) > 20: # 如果这个组合的商品数量超过货架容量（20）
#         print("Warning:项集中的SKU数量超过货架容量(20)")
#         continue  # 忽略超过货架容量的项集（无法放入一个货架）

#     # 检查SKU是否已被分配
#     if not any(sku in used_skus for sku in items):
#         # 如果所有SKU都是全新的：
#         if racks: # 如果已有货架
#             # 现有货架的剩余容量
#             remaining_space = 20 - len(racks[-1]) # 最后一个货架的剩余空间
#             if len(items) <= remaining_space: # 如果剩余空间足够
#                 racks[-1].extend(items) # 将商品加入最后一个货架
#                 used_skus.update(items)  # 标记这些SKU为已分配
#             else:
#                 # 创建新货架
#                 racks.append(items)
#                 used_skus.update(items)
#         else:
#             # 第一个货架
#             racks.append(items)
#             used_skus.update(items)

# # 4. 处理剩余低频SKU
# all_skus = set(df['mat_code'].unique()) # 获取所有SKU
# remaining_skus = all_skus - used_skus # 未被分配的SKU,集合相减

# # 4.将剩余SKU分配到现有货架的空位或新建货架
# for sku in remaining_skus:
#     placed = False
#     for rack in racks:
#         if len(rack) < 20: # 遍历所有货架
#             rack.append(sku) # 找到第一个有空位的货架
#             placed = True
#             break
#     if not placed:
#         racks.append([sku]) # 没有空位则创建新货架

# #5. 输出货架分配方案
# for i, rack in enumerate(racks):
#     print(f"货架{i + 1}: {len(rack)}:{rack}")  # 确保每个货架不超过20个SKU

# #6. 计算优化后的总货架访问次数：
# total_visits = 0
# for order in transaction_data: # 遍历所有订单
#     visits = 0
#     for rack in racks: # 检查每个货架是否包含订单中的SKU
#         if any(sku in rack for sku in order):
#             visits += 1
#         else:
#             visits += 2
#     total_visits += visits
# print(f"总货架访问次数: {total_visits}")

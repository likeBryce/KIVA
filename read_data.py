import pandas as pd
import numpy as np

def Read_data(outbound_path, inventory_path, date_range, order_range):
    # --------------------------
    # 1. 读取outbound数据并生成N和a_in矩阵
    # --------------------------
    # 读取outbound数据
    outbound_df = pd.read_excel(outbound_path)

    # 筛选7月23日数据并取前200条
    outbound_0723 = outbound_df[outbound_df['date_crt'].astype(str).str.startswith(date_range.strip())].head(order_range)

    # 提取唯一订单和商品列表
    unique_orders = outbound_0723['order_num'].unique()
    unique_items_outbound = outbound_0723['mat_code'].unique()


    # 构建a_in矩阵
    N = len(unique_orders)
    P_out = len(unique_items_outbound)
    a_in = np.zeros((P_out, N), dtype=int)

    # 创建映射字典
    order_id_map = {order: idx for idx, order in enumerate(unique_orders)}
    item_id_map = {item: idx for idx, item in enumerate(unique_items_outbound)}

    # 填充矩阵
    for _, row in outbound_0723.iterrows():
        order_idx = order_id_map[row['order_num']]
        item_idx = item_id_map[row['mat_code']]
        a_in[item_idx, order_idx] = 1

    # --------------------------
    # 2. 读取inventory数据并生成M和x_im矩阵
    # --------------------------
    # 读取inventory数据
    inventory_0723 = pd.read_excel(inventory_path)

    # 提取唯一商品和货架列表
    unique_items_inventory = inventory_0723['mat_code'].unique()

    # 只保留bin_code列前6位数字作为货架的唯一标识
    unique_bin = inventory_0723['bin_code'].str[:6]
    unique_bin = unique_bin.unique()

    # 构建x_im矩阵
    M = len(unique_bin)
    P_inven = len(unique_items_inventory)
    x_im = np.zeros((P_out, M), dtype=int)

    # 创建映射字典
    item_id_map = {item: idx for idx, item in enumerate(unique_items_outbound)} # 使用的是出库订单中的所有商品id
    bin_id_map = {bin: idx for idx, bin in enumerate(unique_bin)}

    # 填充矩阵
    for _, row in inventory_0723.iterrows():
        try:
            item_idx = item_id_map[row['mat_code']]
            bin = bin_id_map[row['bin_code'][:6]]
            x_im[item_idx, bin] = 1
        except KeyError: # 只统计出库订单中商品中的货架id
            continue
    return  N, a_in, M, x_im, P_out

if __name__ == '__main__':
    outbound_path = r'.\data\outbound7-23.xlsx'
    inventory_path = r'.\data\inventory7-23.xlsx'
    # 读取7月23日，前200条订单数据
    date_range = "2019-07-23"
    order_range = 200
    N, a_in, M, x_im, P = Read_data(outbound_path, inventory_path, date_range, order_range)
    print("读取数据成功!")


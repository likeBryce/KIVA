#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试基于关联性的SKU分配算法
根据Algorithm 2实现的货架分配策略测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shelf_allocation_strategy_json import ShelfAllocationStrategyJSON
import pandas as pd
import numpy as np

def test_correlation_allocation():
    """
    测试基于关联性的SKU分配算法
    """
    print("=== 测试基于关联性的SKU分配算法 ===")
    
    # 初始化策略
    strategy = ShelfAllocationStrategyJSON()
    
    # 加载数据
    print("\n1. 加载数据...")
    try:
        strategy.load_data_from_json('result/network_analysis_data.json')
        print(f"成功加载数据：{len(strategy.skus)} 个SKU")
        print(f"连接的SKU数量：{len(strategy.connected_skus)}")
    except Exception as e:
        print(f"加载数据失败：{e}")
        return
    
    # 执行DBSCAN聚类
    print("\n2. 执行DBSCAN聚类...")
    clustering_result = strategy.perform_dbscan_clustering(eps=0.8, min_samples=3)
    print(f"聚类结果：{len(strategy.clusters)} 个聚类，{len(strategy.noise_points)} 个噪声点")
    
    # 测试基于关联性的分配
    print("\n3. 测试基于关联性的分配算法...")
    num_shelves = 20
    M = 20  # 每个货架最大容量
    
    shelf_allocation = strategy.allocate_skus_to_shelves(
        num_shelves=num_shelves,
        M=M,
        allocation_method='correlation_based'
    )
    
    # 分析分配结果
    print("\n4. 分配结果分析：")
    total_skus = 0
    non_empty_shelves = 0
    
    for shelf_id, skus in shelf_allocation.items():
        if skus:
            non_empty_shelves += 1
            total_skus += len(skus)
            print(f"货架 {shelf_id}: {len(skus)} 个SKU")
            if len(skus) <= 5:  # 只显示少量SKU的详细信息
                print(f"  SKU列表: {skus}")
            else:
                print(f"  SKU列表: {skus[:3]} ... {skus[-2:]} (显示前3个和后2个)")
    
    print(f"\n总结：")
    print(f"- 使用货架数量：{non_empty_shelves}/{num_shelves}")
    print(f"- 分配SKU总数：{total_skus}")
    print(f"- 平均每货架SKU数：{total_skus/non_empty_shelves:.1f}")
    print(f"- 货架利用率：{total_skus/(non_empty_shelves*M)*100:.1f}%")
    
    # 验证分配约束
    print("\n5. 验证分配约束：")
    max_capacity_violated = False
    for shelf_id, skus in shelf_allocation.items():
        if len(skus) > M:
            print(f"⚠️ 货架 {shelf_id} 超出容量限制：{len(skus)} > {M}")
            max_capacity_violated = True
    
    if not max_capacity_violated:
        print("✅ 所有货架都满足容量约束")
    
    # 检查SKU重复分配
    all_allocated_skus = []
    for skus in shelf_allocation.values():
        all_allocated_skus.extend(skus)
    
    if len(all_allocated_skus) == len(set(all_allocated_skus)):
        print("✅ 没有SKU重复分配")
    else:
        print(f"⚠️ 发现重复分配的SKU：{len(all_allocated_skus) - len(set(all_allocated_skus))} 个")
    
    # 比较不同分配方法
    print("\n6. 比较不同分配方法：")
    
    # 基于聚类的分配
    cluster_allocation = strategy.allocate_skus_to_shelves(
        num_shelves=num_shelves,
        allocation_method='cluster_based'
    )
    
    cluster_total = sum(len(skus) for skus in cluster_allocation.values())
    cluster_shelves = sum(1 for skus in cluster_allocation.values() if skus)
    
    print(f"关联性分配：{total_skus} 个SKU，{non_empty_shelves} 个货架")
    print(f"聚类分配：{cluster_total} 个SKU，{cluster_shelves} 个货架")
    
    return shelf_allocation

def analyze_shelf_correlation(strategy, shelf_allocation):
    """
    分析货架内SKU的关联性
    """
    print("\n=== 分析货架内SKU关联性 ===")
    
    if not hasattr(strategy, 'association_matrix') or strategy.association_matrix is None:
        print("缺少关联矩阵，无法分析关联性")
        return
    
    sku_to_idx = {sku: i for i, sku in enumerate(strategy.skus)}
    
    for shelf_id, skus in shelf_allocation.items():
        if len(skus) > 1:  # 只分析有多个SKU的货架
            correlations = []
            for i, sku1 in enumerate(skus):
                for j, sku2 in enumerate(skus[i+1:], i+1):
                    if sku1 in sku_to_idx and sku2 in sku_to_idx:
                        idx1 = sku_to_idx[sku1]
                        idx2 = sku_to_idx[sku2]
                        corr = strategy.association_matrix[idx1][idx2]
                        correlations.append(corr)
            
            if correlations:
                avg_corr = np.mean(correlations)
                max_corr = np.max(correlations)
                print(f"货架 {shelf_id} ({len(skus)} SKUs): 平均关联度={avg_corr:.3f}, 最大关联度={max_corr:.3f}")

if __name__ == "__main__":
    # 运行测试
    shelf_allocation = test_correlation_allocation()
    
    # 如果测试成功，进行关联性分析
    if shelf_allocation:
        try:
            strategy = ShelfAllocationStrategyJSON()
            strategy.load_data_from_json('result/network_analysis_data.json')
            analyze_shelf_correlation(strategy, shelf_allocation)
        except Exception as e:
            print(f"关联性分析失败：{e}")
    
    print("\n测试完成！")
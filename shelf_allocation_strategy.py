# -*- coding: utf-8 -*-
"""
Version1
货架分配策略实现
基于商品关联网络的DBSCAN聚类和货架分配算法
"""

import pandas as pd
import numpy as np
import json
import copy
import math
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ShelfAllocationStrategyJSON:
    """
    基于JSON数据的货架分配策略类
    从JSON文件加载网络分析结果, 实现DBSCAN聚类和SKUs-to-cluster分配算法
    """
    
    def __init__(self, json_path="d:\\Mycode\\KIVA\\result\\problem1\\analysis_results.json"):
        """
        初始化货架分配策略
        
        Args:
            json_path: JSON文件路径, 包含网络分析结果
        """  

        # 聚类相关属性
        self.clusters = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.noise_points = None
        
        # 货架分配相关属性
        self.x_i = {}
        self.U_r = 6
        self.association_matrix = None
        self.shelf_allocation = None
        self.allocation_strategy = None

        self.json_path = json_path
        self.load_data_from_json()

    # 从JSON文件加载数据
    def load_data_from_json(self):
        """从JSON文件加载数据"""
        print(f"\nStep0: 正在从JSON文件加载数据: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取数据
        self.skus = data['skus']
        self.association_matrix = data['association_matrix']
        self.analysis_results = data['analysis_results']
        
    # 从网络中提取商品特征用于聚类
    def extract_network_features(self):
        """
        从网络中提取商品特征用于聚类
        
        Returns:
            feature_matrix: 特征矩阵
            feature_names: 特征名称列表
        """
        # print("正在提取网络特征...")
        
        features = []
        feature_names = [
            # 'degree',           # 度
            'weighted_degree',  # 加权度
            'sku_sales', # 销量
            'sku_purchase_freq', # 购买频率
            # 'sku_hot_sale_inx' # 长短期销量
        ]
        
        # 计算网络指标
        degrees = self.analysis_results.get('degrees', {}) # 统计度
        weighted_degrees = self.analysis_results.get('weighted_degrees', {}) # 统计加权度:将节点连接的每条边的权重（Weight）相加得到的总和。
        
        # 获取销量, 购买频次, 长短期畅销指数
        sku_sales = self.analysis_results.get('sku_sales', {})
        sku_purchase_freq = self.analysis_results.get('sku_purchase_freq', {})
        sku_hot_sale_inx = self.analysis_results.get('sku_hot_sale_inx', {})
        # abc_classification = self.analysis_results.get('abc_classification', {})
        
        # 只为存在连边的节点构建特征向量
        connected_skus = []  # 存储有连边的SKU
        for sku in self.skus:
            # 检查SKU是否在网络图中有连边（度大于0）
            if degrees.get(sku, 0) > 0:
                feature_vector = [
                    # degrees.get(sku, 0),
                    weighted_degrees.get(sku, 0),
                    sku_sales.get(sku, 0),
                    sku_purchase_freq.get(sku, 0),
                    # sku_hot_sale_inx.get(sku, 0)
                ]
                features.append(feature_vector)
                connected_skus.append(sku)
        
        # 更新SKU列表为只包含有连边的SKU
        self.connected_skus = connected_skus
        feature_matrix = np.array(features)
        print(f"特征提取完成，特征矩阵形状: {feature_matrix.shape}")
        
        return feature_matrix, feature_names

    # 执行DBSCAN聚类 (Algorithm 1)
    def perform_dbscan_clustering(self, eps=0.5, min_samples=5, normalize=True):
        """
        执行DBSCAN聚类
        
        Args:
            eps: DBSCAN的邻域半径参数
            min_samples: 核心点的最小邻居数
            normalize: 是否标准化特征
        
        Returns:
            cluster_labels: 聚类标签
        """
        print(f"\nStep2：正在执行DBSCAN聚类 (邻域半径={eps}, 核心点的最小邻居数={min_samples})...")
        
        # 提取特征
        feature_matrix, feature_names = self.extract_network_features()
        feature_matrix_copy = copy.deepcopy(feature_matrix)
        
        # 特征标准化
        if normalize:
            scaler = StandardScaler()
            normalized_feature_matrix = scaler.fit_transform(feature_matrix) #fit(拟合)：计算每个特征列的均值和标准差, transform(转换)：应用标准化公式转换所有数据
        
        # 执行DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = dbscan.fit_predict(normalized_feature_matrix)
        
        # 保存聚类结果
        self.cluster_labels = cluster_labels
        feature_matrix_labels = np.hstack((feature_matrix_copy, cluster_labels.reshape(-1, 1)))
        df_feature_matrix_labels = pd.DataFrame(feature_matrix_labels, columns=feature_names + ['cluster_label'])
        df_feature_matrix_labels['Average'] = df_feature_matrix_labels[feature_names].mean(axis=1)
        df_feature_matrix_labels.to_excel('d:\\Mycode\\KIVA\\result\\problem1\\dbscan_clustering.xlsx', index=False)
        
        # 分析聚类结果
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"聚类完成！,聚类数量: {n_clusters}, 噪声点数量: {n_noise}, 有效聚类的商品数量: {len(cluster_labels) - n_noise}")
        
        # 计算轮廓系数（如果有多个聚类）
        if n_clusters > 1:
            # 只对非噪声点计算轮廓系数
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(
                    feature_matrix[non_noise_mask], 
                    cluster_labels[non_noise_mask]
                )
                print(f"平均轮廓系数: {silhouette_avg:.3f}")
        
        # 保存聚类统计信息
        self.clusters = self._analyze_clusters()
        
        return cluster_labels
   
    # 记录聚类结果/计算x_i
    def _analyze_clusters(self):
        """
        分析聚类结果
        
        Returns:
            clusters_info: 聚类信息字典
        """
        clusters_info = {}
        
        # 按聚类分组（只处理有连边的SKU）
        for i, sku in enumerate(self.connected_skus):
            cluster_id = self.cluster_labels[i]
            
            if cluster_id not in clusters_info:
                clusters_info[cluster_id] = {
                    'skus': [],
                    'size': 0,
                    'avg_degree': 0,
                    'avg_weighted_degree': 0,
                    'total_sales': 0,
                    'abc_distribution': {'A': 0, 'B': 0, 'C': 0}
                }
            
            clusters_info[cluster_id]['skus'].append(sku)
            clusters_info[cluster_id]['size'] += 1
        
        # 计算聚类统计信息
        degrees = self.analysis_results.get('degrees', {}) 
        weighted_degrees = self.analysis_results.get('weighted_degrees', {}) 
        sku_sales = self.analysis_results.get('sku_sales', {})
        abc_classification = self.analysis_results.get('abc_classification', {})
        
        for cluster_id, info in clusters_info.items():
            # if cluster_id == -1:  # 保留噪声点
            #     continue

            # 计算平均度和加权度
            cluster_degrees = [degrees.get(sku, 0) for sku in info['skus']]
            cluster_weighted_degrees = [weighted_degrees.get(sku, 0) for sku in info['skus']]
            
            info['avg_degree'] = np.mean(cluster_degrees)
            info['avg_weighted_degree'] = np.mean(cluster_weighted_degrees)
            
            # 计算总销量
            info['total_sales'] = sum(sku_sales.get(sku, 0) for sku in info['skus'])
            
            # 计算ABC分布
            for sku in info['skus']:
                if sku in abc_classification.get('A', []):
                    info['abc_distribution']['A'] += 1
                elif sku in abc_classification.get('B', []):
                    info['abc_distribution']['B'] += 1
                elif sku in abc_classification.get('C', []):
                    info['abc_distribution']['C'] += 1
        
        # 计算每个节点的x_i
        # 按照加权度对clusters_info进行降序排序
        sorted_clusters = sorted(clusters_info.items(), key=lambda x: x[1]['avg_weighted_degree'], reverse=True)
        clusters_info = dict(sorted_clusters)

        # 按照加权度对clusters_info分配x_i
        for cluster_id, info in clusters_info.items():
            for sku in info['skus']:
                self.x_i[sku] = self.U_r
            self.U_r -= 2
            if self.U_r == 0:
                break
        # 默认其他节点的x_i == 1
        for sku in set(self.skus) - set(self.connected_skus):
            self.x_i[sku] = 1
    
        return clusters_info

    # 优化参数  
    def optimize_clustering_parameters(self, eps_range=(0.01, 10), min_samples_range=(5, 20)):
        """
        优化DBSCAN聚类参数
        
        Args:
            eps_range: eps参数搜索范围
            min_samples_range: min_samples参数搜索范围
        
        Returns:
            best_params: 最优参数
        """
        print("\nStep1:正在优化聚类参数...")
        # 1. 特征提取
        feature_matrix, feature_names = self.extract_network_features() # 提取特征
        df_feature_matrix = pd.DataFrame(feature_matrix, columns=feature_names) #打印特征
        df_feature_matrix.to_excel(r"D:\Mycode\KIVA\result\problem1\feature_matrix.xlsx")

        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix) # 标准化
        df_feature_matrix = pd.DataFrame(feature_matrix, columns=feature_names) #打印标准化后的特征
        df_feature_matrix.to_excel(r"D:\Mycode\KIVA\result\problem1\normalized_feature_matrix.xlsx")
        print(f"特征均值: {np.mean(feature_matrix)} 特征标准差: {np.std(feature_matrix)}")
        best_score = -1
        best_params = None
        results = []


        # 参数网格搜索
        eps_values = np.linspace(eps_range[0], eps_range[1], 100)
        min_samples_values = range(min_samples_range[0], min_samples_range[1], 1)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                labels = dbscan.fit_predict(feature_matrix)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # 跳过无效聚类结果
                if n_clusters < 2 or n_noise > len(labels) * 0.5: # 噪声点超过一半以上
                    continue
                
                # 计算多个聚类评价指标
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    # 轮廓系数 (Silhouette Score) - 越大越好
                    silhouette = silhouette_score(feature_matrix[non_noise_mask], labels[non_noise_mask])
                    
                    # Calinski-Harabasz指数 (CH指数) - 越大越好
                    ch_score = calinski_harabasz_score(feature_matrix[non_noise_mask], labels[non_noise_mask])
                    
                    # Davies-Bouldin指数 (DB指数) - 越小越好
                    db_score = davies_bouldin_score(feature_matrix[non_noise_mask], labels[non_noise_mask])
                    
                    # 综合评价指标：加权组合
                    # 标准化各指标到[0,1]范围，然后加权平均
                    # 轮廓系数范围[-1,1]，转换为[0,1]
                    silhouette_norm = (silhouette + 1) / 2
                    
                    # CH指数标准化（使用min-max标准化的近似）
                    ch_norm = min(ch_score / 1000, 1.0)  # 假设1000为较好的CH值
                    
                    # DB指数标准化（取倒数并限制范围）
                    db_norm = 1 / (1 + db_score)  # DB越小越好，转换为越大越好
                    
                    # 综合评分（可调整权重）
                    w1, w2, w3 = 0.4, 0.3, 0.3  # 轮廓系数、CH指数、DB指数权重
                    composite_score = w1 * silhouette_norm + w2 * ch_norm + w3 * db_norm
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette_score': silhouette,
                        'ch_score': ch_score,
                        'db_score': db_score,
                        'composite_score': composite_score
                    })
                    
                    # 使用综合评分选择最优参数
                    if composite_score > best_score:
                        best_score = composite_score
                        best_params = {'eps': eps, 'min_samples': min_samples}
        
        # 显示最优参数的详细评价指标
        if best_params and results:
            best_result = next(r for r in results if r['eps'] == best_params['eps'] and r['min_samples'] == best_params['min_samples'])
            print(f"参数优化完成！最优参数: {best_params}")
            print(f"综合评分: {best_result['composite_score']:.3f}")
            print(f"  - 轮廓系数: {best_result['silhouette_score']:.3f}")
            print(f"  - CH指数: {best_result['ch_score']:.2f}")
            print(f"  - DB指数: {best_result['db_score']:.3f}")
            print(f"  - 聚类数: {best_result['n_clusters']}, 噪声点: {best_result['n_noise']}")
        
        return best_params, results
    
    # 分析和可视化聚类评价指标
    def analyze_clustering_metrics(self, results, top_n=10):
        """
        分析和可视化聚类评价指标
        
        Args:
            results: optimize_clustering_parameters返回的结果列表
            top_n: 显示前N个最优结果
        """
        if not results:
            print("没有可用的聚类结果进行分析")
            return
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(results)
        
        # 按综合评分排序
        df_sorted = df.sort_values('composite_score', ascending=False)
        
        print(f"\n=== 聚类评价指标综合分析 ===")
        print(f"总共测试了 {len(results)} 组参数组合")
        print(f"\n前{top_n}个最优参数组合:")
        print("-" * 100)
        print(f"{'排名':<4} {'eps':<8} {'min_samples':<12} {'聚类数':<6} {'噪声点':<8} {'轮廓系数':<10} {'CH指数':<10} {'DB指数':<10} {'综合评分':<10}")
        print("-" * 100)
        
        for i, (_, row) in enumerate(df_sorted.head(top_n).iterrows()):
            print(f"{i+1:<4}   {row['eps']:<8.3f}   {row['min_samples']:<12}   {row['n_clusters']:<6}   {row['n_noise']:<8}"
                  f"{  row['silhouette_score']:<10.3f}   {row['ch_score']:<10.2f}   {row['db_score']:<10.3f}   {row['composite_score']:<10.3f}")
        
        # 统计分析
        print(f"\n=== 评价指标统计分析 ===")
        print(f"轮廓系数 - 均值: {df['silhouette_score'].mean():.3f}, 标准差: {df['silhouette_score'].std():.3f}, 范围: [{df['silhouette_score'].min():.3f}, {df['silhouette_score'].max():.3f}]")
        print(f"CH指数    - 均值: {df['ch_score'].mean():.2f}, 标准差: {df['ch_score'].std():.2f}, 范围: [{df['ch_score'].min():.2f}, {df['ch_score'].max():.2f}]")
        print(f"DB指数    - 均值: {df['db_score'].mean():.3f}, 标准差: {df['db_score'].std():.3f}, 范围: [{df['db_score'].min():.3f}, {df['db_score'].max():.3f}]")
        print(f"综合评分  - 均值: {df['composite_score'].mean():.3f}, 标准差: {df['composite_score'].std():.3f}, 范围: [{df['composite_score'].min():.3f}, {df['composite_score'].max():.3f}]")
        
        # 可视化对比
        self.plot_clustering_metrics_comparison(df)
        
        return df_sorted
    
    # 绘制聚类评价指标对比图
    def plot_clustering_metrics_comparison(self, df):
        """
        绘制聚类评价指标对比图
        
        Args:
            df: 包含评价指标的DataFrame
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('聚类评价指标综合对比分析', fontsize=16, fontweight='bold')
        
        # 1. 各指标分布直方图
        axes[0, 0].hist(df['silhouette_score'], bins=20, alpha=0.7, color='skyblue', label='轮廓系数')
        axes[0, 0].set_title('轮廓系数分布')
        axes[0, 0].set_xlabel('轮廓系数')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. CH指数 vs DB指数散点图
        scatter = axes[0, 1].scatter(df['ch_score'], df['db_score'], c=df['composite_score'], 
                                   cmap='viridis', alpha=0.6, s=50)
        axes[0, 1].set_title('CH指数 vs DB指数 (颜色表示综合评分)')
        axes[0, 1].set_xlabel('CH指数')
        axes[0, 1].set_ylabel('DB指数')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='综合评分')
        
        # 3. 轮廓系数 vs 综合评分
        axes[1, 0].scatter(df['silhouette_score'], df['composite_score'], alpha=0.6, color='orange')
        axes[1, 0].set_title('轮廓系数 vs 综合评分')
        axes[1, 0].set_xlabel('轮廓系数')
        axes[1, 0].set_ylabel('综合评分')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 聚类数量分布
        cluster_counts = df['n_clusters'].value_counts().sort_index()
        axes[1, 1].bar(cluster_counts.index, cluster_counts.values, alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('聚类数量分布')
        axes[1, 1].set_xlabel('聚类数量')
        axes[1, 1].set_ylabel('参数组合数量')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 保存图片
        plt.savefig('clustering_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("\n聚类评价指标对比图已保存为 'clustering_metrics_comparison.png'")

    # 将SKU分配到货架主代码
    def allocate_skus_to_shelves(self, num_shelves=408, M=20, allocation_method='correlation_based'):
        """
        将SKU分配到货架
        
        Args:
            num_shelves: 货架数量
            M: 每个货架的最大SKU容量
            allocation_method: 分配方法 ('correlation_based', 'cluster_based')
        
        Returns:
            shelf_allocation: 货架分配结果
        """
        print(f"正在执行SKU到货架的分配...")
        
        if self.cluster_labels is None:
            print("请先执行聚类分析！")
            return None
        
        if allocation_method == 'correlation_based':
            shelf_allocation = self._correlation_based_allocation(num_shelves, M)
        
        self.shelf_allocation = shelf_allocation
        
        # 计算分配统计信息
        self._calculate_allocation_stats()
    
        return shelf_allocation
    
    # 基于关联性的SKU到货架分配算法 (Algorithm 2)
    def _correlation_based_allocation(self, num_shelves, M):
        """
        基于关联性的SKU到货架分配算法 (Algorithm 2)
        根据论文中的伪代码实现
        
        Args:
            num_shelves: 最大货架数量
            M: 每个货架的最大SKU容量
        
        Returns:
            shelf_allocation: 货架分配结果 y_i,c
        """
        print(f"\nStep3: 执行基于关联性的分配算法，可用货架数量: {num_shelves}, 容量限制: M={M}")
        
        # 初始化：数据处理
        R =  math.ceil(sum(int(self.x_i[sku]) for sku in self.skus) // M) # 实际需要的最大货架数量
        R = min(R, num_shelves) # 确保不超过最大货架数量
        shelf_allocation = {c: [] for c in range(R)}  # y_i,c = 0 for all i,c
        
        # 创建SKU到索引的映射
        sku_to_idx = {sku: idx for idx, sku in enumerate(self.skus)}
        idx_to_sku = {i: sku for i, sku in enumerate(self.skus)}
        
        # 计算每个SKU的权重 w_i (使用度中心性)
        weights = self.analysis_results['weighted_degrees']
        
        # 可用SKU集合 I
        available_skus = {sku:int(self.x_i[sku]) for sku in self.skus}

        # 已用SKU集合 I: used_skus[sku]
        used_skus = []
        
        # Algorithm 2: SKUs-to-cluster assignment based on correlation
        for c in range(R):  # for c = 1 to |R| do
            if not available_skus:
                break
                
            # Step 4: initial clustering center i ← max(w_i)
            center_sku = max(available_skus.keys(), key=lambda i: weights[i])
            
            # Step 5: y_i,c ← y_i,c + 1
            shelf_allocation[c].append(center_sku)
            
            # Step 6: x_i ← x_i - 1 (从可用集合中移除)
            available_skus[center_sku] -= 1
            
            # Step 7-9: 如果SKU用完了，从I中删除
            if available_skus[center_sku] == 0:
                del available_skus[center_sku] #???
                used_skus.append(center_sku)
            
            # Step 10-19: while |c| ≤ M do
            while len(shelf_allocation[c]) < M and available_skus:
                # Step 11: selecting SKU j ← max(Σ_{i∈c} s_{i,j})
                best_sku_idx = None
                max_correlation = -1

                # 计算与当前货架中所有SKU的关联度总和
                for sku in set(available_skus.keys()) - set(used_skus):
                    total_correlation = 0
                    sku_idx = sku_to_idx[sku]
                    for allocated_sku in shelf_allocation[c]:
                        allocated_idx = sku_to_idx[allocated_sku]
                        total_correlation += self.association_matrix[allocated_idx][sku_idx]
                    
                    if total_correlation > max_correlation:
                        max_correlation = total_correlation
                        best_sku_idx = sku_idx
                
                # Step 12-13: 如果最大关联度为0，选择与货架上已有SKU销量相似度最大的SKU
                if max_correlation == 0 and available_skus:
                    # selecting SKU j ← min(Σ_{i∈c} √(s_{j,j} - s_{i,j})²)
                    min_distance = float('inf')
                    for j_sku in available_skus.keys():
                        total_distance = 0
                        j_idx = sku_to_idx[j_sku]
                        for allocated_sku in shelf_allocation[c]:
                            allocated_idx = sku_to_idx[allocated_sku]
                            # 简化距离计算
                            distance = abs(self.association_matrix[j_idx][j_idx] - self.association_matrix[allocated_idx][j_idx])
                            total_distance += distance
                        
                        if total_distance < min_distance:
                            min_distance = total_distance
                            best_sku_idx = j_idx
                
                if best_sku_idx is not None:
                    # Step 14: add j to c
                    # Step 15: y_j,c ← y_j,c + 1
                    best_sku = idx_to_sku[best_sku_idx]
                    shelf_allocation[c].append(best_sku)
                    
                    # Step 16: x_j ← x_j - 1
                    available_skus[best_sku] -= 1
                    
                    # Step 17-19: 如果SKU用完了，从I中删除
                    if available_skus[best_sku] == 0:
                        del available_skus[best_sku]
                        used_skus.append(best_sku)
                else:
                    break
                
                # Step 20-21: if |I| == 0 then break
                if not available_skus:
                    break
        
        print(f"分配完成！使用了 {R} 个货架，平均每个货架 {sum(len(skus) for skus in shelf_allocation.values()) / R:.1f} 个SKU")
        
        return shelf_allocation

    # 计算分配统计信息
    def _calculate_allocation_stats(self):
        """
        计算分配统计信息
        """
        if not self.shelf_allocation:
            return
        
        stats = {
            'total_shelves': len(self.shelf_allocation),
            'used_shelves': sum(1 for shelf_skus in self.shelf_allocation.values() if shelf_skus),
            'avg_skus_per_shelf': np.mean([len(skus) for skus in self.shelf_allocation.values()]),
            'max_skus_per_shelf': max(len(skus) for skus in self.shelf_allocation.values()),
            'min_skus_per_shelf': min(len(skus) for skus in self.shelf_allocation.values() if skus),
            'empty_shelves': sum(1 for shelf_skus in self.shelf_allocation.values() if not shelf_skus)
        }
        
        self.allocation_stats = stats
        
        print(f"分配统计信息:")
        print(f"  总货架数: {stats['total_shelves']}")
        print(f"  使用货架数: {stats['used_shelves']}")
        print(f"  平均每货架SKU数: {stats['avg_skus_per_shelf']:.2f}")
        print(f"  最大每货架SKU数: {stats['max_skus_per_shelf']}")
        print(f"  最小每货架SKU数: {stats['min_skus_per_shelf']}")
        print(f"  空货架数: {stats['empty_shelves']}")

    # 可视化聚类结果
    def visualize_clustering_results(self, save_path="d:\\Mycode\\KIVA\\result\\problem1\\clustering_results.png"):
        """
        可视化聚类结果
        """
        if self.cluster_labels is None:
            print("请先执行聚类分析！")
            return
        
        print("正在生成聚类结果可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 聚类节点空间分布图
        if hasattr(self, 'connected_skus') and hasattr(self, 'cluster_labels'):
            # 创建子图用于显示聚类的节点
            subgraph_nodes = self.connected_skus
            subgraph = self.G.subgraph(subgraph_nodes)
            
            # 使用网络布局算法
            try:
                if len(subgraph_nodes) <= 50:
                    pos = nx.kamada_kawai_layout(subgraph)
                else:
                    pos = nx.spring_layout(subgraph, k=1, iterations=50)
            except:
                pos = nx.circular_layout(subgraph)
            
            # 为每个聚类分配颜色
            unique_labels = set(self.cluster_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            # 绘制节点，按聚类着色
            for i, sku in enumerate(subgraph_nodes):
                cluster_id = self.cluster_labels[i]
                color = color_map[cluster_id]
                
                if cluster_id == -1:  # 噪声点用黑色
                    axes[0, 0].scatter(pos[sku][0], pos[sku][1], 
                                     c='black', s=50, alpha=0.7, marker='x')
                else:
                    axes[0, 0].scatter(pos[sku][0], pos[sku][1], 
                                     c=[color], s=80, alpha=0.8)
            
            # 绘制边
            for edge in subgraph.edges():
                x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                axes[0, 0].plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=0.5)
            
            axes[0, 0].set_title('聚类节点空间分布图')
            axes[0, 0].set_xlabel('X坐标')
            axes[0, 0].set_ylabel('Y坐标')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加图例
            legend_elements = []
            for label in unique_labels:
                if label == -1:
                    legend_elements.append(plt.Line2D([0], [0], marker='x', color='w', 
                                                    markerfacecolor='black', markersize=8, 
                                                    label='噪声点'))
                else:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color_map[label], markersize=8, 
                                                    label=f'聚类{label}'))
            axes[0, 0].legend(handles=legend_elements, loc='upper right')
        else:
            axes[0, 0].text(0.5, 0.5, '无聚类数据', ha='center', va='center', 
                           transform=axes[0, 0].transAxes, fontsize=14)
            axes[0, 0].set_title('聚类节点空间分布图')

        
        # 2. 聚类大小分布
        if valid_clusters:
            axes[0, 1].bar(range(len(valid_clusters)), list(valid_clusters.values()))
            axes[0, 1].set_xlabel('聚类ID')
            axes[0, 1].set_ylabel('SKU数量')
            axes[0, 1].set_title('各聚类大小分布')
            axes[0, 1].set_xticks(range(len(valid_clusters)))
            axes[0, 1].set_xticklabels(list(valid_clusters.keys()))
        
        # 3. 聚类特征分析
        if self.clusters:
            cluster_ids = [cid for cid in self.clusters.keys() if cid != -1]
            avg_degrees = [self.clusters[cid]['avg_degree'] for cid in cluster_ids]
            total_sales = [self.clusters[cid]['total_sales'] for cid in cluster_ids]
            
            axes[1, 0].scatter(avg_degrees, total_sales, 
                             c=range(len(cluster_ids)), cmap='viridis')
            axes[1, 0].set_xlabel('平均度')
            axes[1, 0].set_ylabel('总销量')
            axes[1, 0].set_title('聚类特征分析')
            
            # 添加聚类标签
            for i, cid in enumerate(cluster_ids):
                axes[1, 0].annotate(f'C{cid}', (avg_degrees[i], total_sales[i]))
        
        # 4. ABC分类在聚类中的分布
        if self.clusters:
            cluster_ids = [cid for cid in self.clusters.keys() if cid != -1]
            abc_data = []
            
            for cid in cluster_ids:
                abc_dist = self.clusters[cid]['abc_distribution']
                abc_data.append([abc_dist['A'], abc_dist['B'], abc_dist['C']])
            
            if abc_data:
                abc_array = np.array(abc_data)
                bottom_b = abc_array[:, 0]
                bottom_c = abc_array[:, 0] + abc_array[:, 1]
                
                axes[1, 1].bar(range(len(cluster_ids)), abc_array[:, 0], 
                              label='A类', color='red', alpha=0.7)
                axes[1, 1].bar(range(len(cluster_ids)), abc_array[:, 1], 
                              bottom=bottom_b, label='B类', color='orange', alpha=0.7)
                axes[1, 1].bar(range(len(cluster_ids)), abc_array[:, 2], 
                              bottom=bottom_c, label='C类', color='green', alpha=0.7)
                
                axes[1, 1].set_xlabel('聚类ID')
                axes[1, 1].set_ylabel('SKU数量')
                axes[1, 1].set_title('各聚类中ABC分类分布')
                axes[1, 1].set_xticks(range(len(cluster_ids)))
                axes[1, 1].set_xticklabels([f'C{cid}' for cid in cluster_ids])
                axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"聚类结果可视化已保存到: {save_path}")
    
    def visualize_shelf_allocation(self, save_path="d:\\Mycode\\KIVA\\result\\problem1\\shelf_allocation.png"):
        """
        可视化货架分配结果
        """
        if not self.shelf_allocation:
            print("请先执行货架分配！")
            return
        
        print("正在生成货架分配可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 货架利用率分布
        shelf_sizes = [len(skus) for skus in self.shelf_allocation.values()]
        axes[0, 0].hist(shelf_sizes, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('每货架SKU数量')
        axes[0, 0].set_ylabel('货架数量')
        axes[0, 0].set_title('货架利用率分布')
        axes[0, 0].axvline(np.mean(shelf_sizes), color='red', linestyle='--', 
                          label=f'平均值: {np.mean(shelf_sizes):.1f}')
        axes[0, 0].legend()
        
        # 2. 货架负载热力图（前20个货架）
        top_shelves = min(20, len(self.shelf_allocation))
        shelf_loads = [len(self.shelf_allocation[i]) for i in range(top_shelves)]
        
        # 创建热力图数据
        heatmap_data = np.array(shelf_loads).reshape(4, 5) if top_shelves == 20 else np.array(shelf_loads + [0] * (20 - len(shelf_loads))).reshape(4, 5)
        
        im = axes[0, 1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        axes[0, 1].set_title('货架负载热力图 (前20个货架)')
        axes[0, 1].set_xlabel('货架列')
        axes[0, 1].set_ylabel('货架行')
        
        # 添加数值标注
        for i in range(4):
            for j in range(5):
                text = axes[0, 1].text(j, i, int(heatmap_data[i, j]),
                                     ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=axes[0, 1])
        
        # 3. 聚类在货架中的分布
        if self.cluster_labels is not None:
            cluster_shelf_mapping = defaultdict(list)
            
            for shelf_id, skus in self.shelf_allocation.items():
                for sku in skus:
                    sku_idx = self.skus.index(sku)
                    cluster_id = self.cluster_labels[sku_idx]
                    cluster_shelf_mapping[cluster_id].append(shelf_id)
            
            # 计算每个聚类分布在多少个货架上
            cluster_spread = {cid: len(set(shelves)) for cid, shelves in cluster_shelf_mapping.items()}
            
            if cluster_spread:
                valid_clusters = {k: v for k, v in cluster_spread.items() if k != -1}
                if valid_clusters:
                    axes[1, 0].bar(range(len(valid_clusters)), list(valid_clusters.values()))
                    axes[1, 0].set_xlabel('聚类ID')
                    axes[1, 0].set_ylabel('分布货架数')
                    axes[1, 0].set_title('聚类在货架中的分散程度')
                    axes[1, 0].set_xticks(range(len(valid_clusters)))
                    axes[1, 0].set_xticklabels([f'C{k}' for k in valid_clusters.keys()])
        
        # 4. ABC分类在货架中的分布
        abc_classification = self.analysis_results.get('abc_classification', {})
        shelf_abc_dist = []
        
        for shelf_id, skus in self.shelf_allocation.items():
            abc_count = {'A': 0, 'B': 0, 'C': 0}
            for sku in skus:
                if sku in abc_classification.get('A', []):
                    abc_count['A'] += 1
                elif sku in abc_classification.get('B', []):
                    abc_count['B'] += 1
                elif sku in abc_classification.get('C', []):
                    abc_count['C'] += 1
            shelf_abc_dist.append([abc_count['A'], abc_count['B'], abc_count['C']])
        
        if shelf_abc_dist:
            # 只显示前20个货架
            display_shelves = min(20, len(shelf_abc_dist))
            abc_array = np.array(shelf_abc_dist[:display_shelves])
            
            if abc_array.size > 0:
                bottom_b = abc_array[:, 0]
                bottom_c = abc_array[:, 0] + abc_array[:, 1]
                
                x_pos = range(display_shelves)
                axes[1, 1].bar(x_pos, abc_array[:, 0], label='A类', color='red', alpha=0.7)
                axes[1, 1].bar(x_pos, abc_array[:, 1], bottom=bottom_b, label='B类', color='orange', alpha=0.7)
                axes[1, 1].bar(x_pos, abc_array[:, 2], bottom=bottom_c, label='C类', color='green', alpha=0.7)
                
                axes[1, 1].set_xlabel('货架ID')
                axes[1, 1].set_ylabel('SKU数量')
                axes[1, 1].set_title(f'各货架ABC分类分布 (前{display_shelves}个货架)')
                axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"货架分配可视化已保存到: {save_path}")
    
    def generate_allocation_report(self, output_path="d:\\Mycode\\KIVA\\result\\problem1\\shelf_allocation_report.md"):
        """
        生成货架分配策略报告
        """
        print("正在生成货架分配策略报告...")
        
        report_content = f"""# 货架分配策略分析报告

        ## 1. 执行概述

        本报告基于商品关联网络分析结果，采用DBSCAN聚类算法对商品进行聚类，并实现了智能化的货架分配策略。

        ### 1.1 数据基础
        - 商品总数: {len(self.skus)}
        - 网络节点数: {self.analysis_results.get('network_nodes', 0)}
        - 网络边数: {self.analysis_results.get('network_edges', 0)}
        - 网络密度: {self.analysis_results.get('network_density', 0):.4f}

        ## 2. 聚类分析结果
        """
                
        if self.cluster_labels is not None:
            n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            n_noise = list(self.cluster_labels).count(-1)
            
            report_content += f"""
        ### 2.1 DBSCAN聚类统计
        - 聚类数量: {n_clusters}
        - 噪声点数量: {n_noise}
        - 有效聚类商品数: {len(self.cluster_labels) - n_noise}
        - 聚类覆盖率: {((len(self.cluster_labels) - n_noise) / len(self.cluster_labels) * 100):.1f}%

        ### 2.2 各聚类详细信息
        """
                    
        if self.clusters:
            for cluster_id, info in self.clusters.items():
                if cluster_id == -1:
                    report_content += f"""
        **噪声点聚类 (ID: -1)**
        - 商品数量: {info['size']}
        - 商品列表: {', '.join(info['skus'][:10])}{'...' if len(info['skus']) > 10 else ''}
        """
        else:
            report_content += f"""
        **聚类 {cluster_id}**
        - 商品数量: {info['size']}
        - 平均度: {info['avg_degree']:.2f}
        - 平均加权度: {info['avg_weighted_degree']:.2f}
        - 总销量: {info['total_sales']}
        - ABC分布: A类({info['abc_distribution']['A']}) B类({info['abc_distribution']['B']}) C类({info['abc_distribution']['C']})
        - 代表商品: {', '.join(info['skus'][:5])}{'...' if len(info['skus']) > 5 else ''}

        """
                
        report_content += "\n## 3. 货架分配策略\n"
                
        if self.shelf_allocation:
            report_content += f"""
        ### 3.1 分配策略: {self.allocation_strategy}

        ### 3.2 分配统计信息
        """
                    
        if hasattr(self, 'allocation_stats'):
            stats = self.allocation_stats
            report_content += f"""
        - 总货架数: {stats['total_shelves']}
        - 使用货架数: {stats['used_shelves']}
        - 货架利用率: {(stats['used_shelves'] / stats['total_shelves'] * 100):.1f}%
        - 平均每货架SKU数: {stats['avg_skus_per_shelf']:.2f}
        - 最大每货架SKU数: {stats['max_skus_per_shelf']}
        - 最小每货架SKU数: {stats['min_skus_per_shelf']}
        - 空货架数: {stats['empty_shelves']}
        """
                    
            # 展示部分货架分配结果
            report_content += "\n### 3.3 货架分配示例\n\n"
            
            for shelf_id in sorted(self.shelf_allocation.keys())[:10]:  # 只显示前10个货架
                skus = self.shelf_allocation[shelf_id]
                if skus:
                    report_content += f"**货架 {shelf_id}** ({len(skus)}个SKU): {', '.join(skus[:5])}{'...' if len(skus) > 5 else ''}\n\n"
        
        report_content += f"""
        ## 4. 策略优势分析

        ### 4.1 基于网络聚类的优势
        1. **关联性保持**: 通过DBSCAN聚类，将具有强关联关系的商品分配到相近的货架，减少拣选路径
        2. **销售特征考虑**: 结合ABC分析和销量数据，优先配置高价值商品
        3. **网络拓扑利用**: 利用度中心性、介数中心性等网络指标，识别关键商品节点

        ### 4.2 分配策略特点
        1. **聚类内聚性**: 同一聚类的商品倾向于分配在相邻货架
        2. **负载均衡**: 避免某些货架过载，保持系统整体效率
        3. **动态适应**: 可根据网络变化动态调整分配策略

        ## 5. 实施建议

        ### 5.1 部署策略
        1. **分阶段实施**: 先对A类商品进行优化配置，再逐步扩展到B、C类商品
        2. **效果监控**: 建立关键指标监控体系，跟踪拣选效率改善情况
        3. **定期更新**: 根据销售数据变化，定期重新分析网络并调整配置

        ### 5.2 关键指标
        1. **拣选路径长度**: 监控平均拣选路径是否缩短
        2. **货架访问频率**: 分析货架访问的均衡性
        3. **订单完成时间**: 评估整体拣选效率提升

        ## 6. 技术实现

        ### 6.1 算法流程
        1. 商品关联网络构建
        2. 网络特征提取（度、中心性、销量等）
        3. DBSCAN聚类分析
        4. 聚类结果优化
        5. 货架分配策略执行

        ### 6.2 参数配置
        - DBSCAN eps参数: 控制聚类密度
        - min_samples参数: 控制核心点阈值
        - 货架数量: 根据仓库实际情况配置
        - 分配策略: 支持多种分配方法

        ---

        *报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
        *分析工具: Python + NetworkX + Scikit-learn*
        """
                
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"货架分配策略报告已保存到: {output_path}")
    
    def run_complete_allocation_analysis(self, eps=0.5, min_samples=5, num_shelves=408, 
                                       allocation_method='cluster_based', optimize_params=False):
        """
        运行完整的货架分配分析流程
        
        Args:
            eps: DBSCAN eps参数
            min_samples: DBSCAN min_samples参数
            num_shelves: 货架数量
            allocation_method: 分配方法
            optimize_params: 是否优化聚类参数
        """
        # print("开始完整的货架分配分析流程...")
        
        # 1. 参数优化（可选）
        if optimize_params:
            best_params, results = self.optimize_clustering_parameters()
            if best_params:
                eps = best_params['eps']
                min_samples = best_params['min_samples']
                print(f"使用优化后的参数: eps={eps:.3f}, min_samples={min_samples}")
        
        # 2. 执行聚类
        self.perform_dbscan_clustering(eps=eps, min_samples=min_samples)
        
        # 3. 货架分配
        self.allocate_skus_to_shelves(num_shelves=408, 
                                    allocation_method=allocation_method)
        
        # 4. 生成可视化

        self.analyze_clustering_metrics(results, top_n=10)
        
        self.visualize_clustering_results()
        self.visualize_shelf_allocation()
        
        # # 5. 生成报告
        # self.generate_allocation_report()
        
        
        return {
            'clusters': self.clusters,
            'shelf_allocation': self.shelf_allocation,
            'allocation_stats': getattr(self, 'allocation_stats', None)
        }

# 使用示例
if __name__ == "__main__":

    # 创建基于JSON的货架分配策略
    shelf_strategy = ShelfAllocationStrategyJSON()
    
    # 运行完整的分配分析
    results = shelf_strategy.run_complete_allocation_analysis(
        eps=0.8,
        min_samples=3,
        num_shelves=408,
        allocation_method='correlation_based',
        optimize_params=True  # 启动参数优化
    )
      
    print("\n分析完成!")
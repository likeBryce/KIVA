# -*- coding: utf-8 -*-
"""
货架分配策略实现
基于商品关联网络的DBSCAN聚类和货架分配算法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ShelfAllocationStrategy:
    """
    基于商品关联网络的货架分配策略类
    实现DBSCAN聚类和SKUs-to-cluster分配算法
    """
    
    def __init__(self, network_analyzer):
        """
        初始化货架分配策略
        
        Args:
            network_analyzer: SKUAssociationAnalyzer实例，包含网络分析结果
        """
        self.analyzer = network_analyzer
        self.G = network_analyzer.G
        self.skus = network_analyzer.skus
        self.analysis_results = network_analyzer.analysis_results
        
        # 聚类相关属性
        self.clusters = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.noise_points = None
        
        # 货架分配相关属性
        self.shelf_allocation = None
        self.allocation_strategy = None
        
    def extract_network_features(self):
        """
        从网络中提取商品特征用于聚类
        
        Returns:
            feature_matrix: 特征矩阵
            feature_names: 特征名称列表
        """
        print("正在提取网络特征...")
        
        features = []
        feature_names = [
            'degree',           # 度
            'weighted_degree',  # 加权度
            'clustering_coeff', # 聚类系数
            'betweenness',      # 介数中心性
            'closeness',        # 接近中心性
            'pagerank',         # PageRank值
            'sales_volume',     # 销量
            'abc_score'         # ABC分类得分
        ]
        
        # 计算网络指标
        degrees = dict(self.G.degree()) # 统计度
        weighted_degrees = dict(self.G.degree(weight='weight')) # 统计加权度
        clustering_coeffs = nx.clustering(self.G) # 计算聚类系数
        
        # 处理空图的情况
        if self.G.number_of_nodes() == 0:
            betweenness = {}
            closeness = {}
            pagerank = {}
        else:
            try:
                betweenness = nx.betweenness_centrality(self.G) # 计算介中心性
                closeness = nx.closeness_centrality(self.G) # 计算接近中心性
                # 兼容旧版本scipy，使用更稳定的PageRank实现
                try:
                    # PageRank 是 Google 创始人拉里·佩奇和谢尔盖·布林开发的算法，用于衡量网页的重要性。在图论中，它衡量图中节点的重要性或中心性：
                    pagerank = nx.pagerank(self.G, max_iter=100) # 计算PageRank(网页排序)值
                except (AttributeError, ImportError):
                    # 如果PageRank失败，使用度中心性作为替代
                    print("警告: PageRank计算失败，使用度中心性作为替代")
                    total_degree = sum(degrees.values()) if degrees else 1
                    pagerank = {node: degree/total_degree for node, degree in degrees.items()}
            except Exception as e:
                print(f"警告: 网络指标计算出现问题: {e}，使用默认值")
                betweenness = {node: 0 for node in self.G.nodes()}
                closeness = {node: 0 for node in self.G.nodes()}
                pagerank = {node: 1/len(self.skus) if self.skus else 0 for node in self.G.nodes()}
        
        # 获取销量和ABC分类信息
        sku_sales = self.analysis_results.get('sku_sales', {})
        abc_classification = self.analysis_results.get('abc_classification', {})
        
        # 为每个SKU构建特征向量
        for sku in self.skus:
            # ABC分类得分
            if sku in abc_classification.get('A', []):
                abc_score = 3
            elif sku in abc_classification.get('B', []):
                abc_score = 2
            elif sku in abc_classification.get('C', []):
                abc_score = 1
            else:
                abc_score = 0
            
            feature_vector = [
                degrees.get(sku, 0),
                weighted_degrees.get(sku, 0),
                clustering_coeffs.get(sku, 0),
                betweenness.get(sku, 0),
                closeness.get(sku, 0),
                pagerank.get(sku, 0),
                sku_sales.get(sku, 0),
                abc_score
            ]
            features.append(feature_vector)
        
        feature_matrix = np.array(features)
        print(f"特征提取完成，特征矩阵形状: {feature_matrix.shape}")
        
        return feature_matrix, feature_names
    
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
        print(f"正在执行DBSCAN聚类 (邻域半径={eps}, 核心点的最小邻居数={min_samples})...")
        
        # 提取特征
        feature_matrix, feature_names = self.extract_network_features()
        
        # 特征标准化
        if normalize:
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix) #fit(拟合)：计算每个特征列的均值和标准差, transform(转换)：应用标准化公式转换所有数据
        
        # 执行DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(feature_matrix)
        
        # 保存聚类结果
        self.cluster_labels = cluster_labels
        
        # 分析聚类结果
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"聚类完成！")
        print(f"聚类数量: {n_clusters}")
        print(f"噪声点数量: {n_noise}")
        print(f"有效聚类的商品数量: {len(cluster_labels) - n_noise}")
        
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
    
    def _analyze_clusters(self):
        """
        分析聚类结果
        
        Returns:
            clusters_info: 聚类信息字典
        """
        clusters_info = {}
        
        # 按聚类分组
        for i, sku in enumerate(self.skus):
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
        degrees = dict(self.G.degree())
        weighted_degrees = dict(self.G.degree(weight='weight'))
        sku_sales = self.analysis_results.get('sku_sales', {})
        abc_classification = self.analysis_results.get('abc_classification', {})
        
        for cluster_id, info in clusters_info.items():
            if cluster_id == -1:  # 噪声点
                continue
                
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
        
        return clusters_info

    # 优化DBSCAN聚类参数  
    def optimize_clustering_parameters(self, eps_range=(0.1, 2.0), min_samples_range=(3, 20)):
        """
        优化DBSCAN聚类参数
        
        Args:
            eps_range: eps参数搜索范围
            min_samples_range: min_samples参数搜索范围
        
        Returns:
            best_params: 最优参数
        """
        print("正在优化聚类参数...")
        
        feature_matrix, _ = self.extract_network_features()
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        
        best_score = -1
        best_params = None
        results = []
        
        # 参数网格搜索
        eps_values = np.linspace(eps_range[0], eps_range[1], 10)
        min_samples_values = range(min_samples_range[0], min_samples_range[1], 2)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(feature_matrix)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # 跳过无效聚类结果
                if n_clusters < 2 or n_noise > len(labels) * 0.5: # 噪声点超过一半以上
                    continue
                
                # 计算轮廓系数
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    score = silhouette_score(feature_matrix[non_noise_mask], labels[non_noise_mask])
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette_score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
        
        print(f"参数优化完成！最优参数: {best_params}, 最优轮廓系数: {best_score:.3f}")
        
        return best_params, results
    
    def allocate_skus_to_shelves(self, num_shelves=20, allocation_method='balanced'):
        """
        将SKU分配到货架
        
        Args:
            num_shelves: 货架数量
            allocation_method: 分配方法 ('balanced', 'sales_based', 'cluster_based')
        
        Returns:
            shelf_allocation: 货架分配结果
        """
        print(f"正在执行SKU到货架的分配 (方法: {allocation_method})...")
        
        if self.cluster_labels is None:
            print("请先执行聚类分析！")
            return None
        
        shelf_allocation = {i: [] for i in range(num_shelves)} # 记录每个货架的分配策略
        
        if allocation_method == 'cluster_based':
            shelf_allocation = self._cluster_based_allocation(num_shelves)
        elif allocation_method == 'sales_based':
            shelf_allocation = self._sales_based_allocation(num_shelves)
        else:  # balanced
            shelf_allocation = self._balanced_allocation(num_shelves)
        
        self.shelf_allocation = shelf_allocation
        self.allocation_strategy = allocation_method
        
        # 计算分配统计信息
        self._calculate_allocation_stats()
        
        return shelf_allocation
    
    def _cluster_based_allocation(self, num_shelves):
        """
        基于聚类的分配策略
        """
        shelf_allocation = {i: [] for i in range(num_shelves)}
        
        # 获取有效聚类（排除噪声点）
        valid_clusters = [cid for cid in self.clusters.keys() if cid != -1]
        
        if not valid_clusters:
            print("没有有效的聚类，使用平衡分配策略")
            return self._balanced_allocation(num_shelves)
        
        # 按聚类大小排序
        sorted_clusters = sorted(valid_clusters, 
                               key=lambda x: self.clusters[x]['size'], 
                               reverse=True)
        
        # 为每个聚类分配货架
        shelves_per_cluster = max(1, num_shelves // len(sorted_clusters))
        current_shelf = 0
        
        for cluster_id in sorted_clusters:
            cluster_skus = self.clusters[cluster_id]['skus']
            
            # 计算该聚类需要的货架数
            cluster_shelves = min(shelves_per_cluster, 
                                max(1, len(cluster_skus) // 10))  # 每个货架最多10个SKU
            
            # 将聚类内的SKU分配到指定货架
            skus_per_shelf = len(cluster_skus) // cluster_shelves
            
            for i in range(cluster_shelves):
                if current_shelf >= num_shelves:
                    break
                    
                start_idx = i * skus_per_shelf
                end_idx = start_idx + skus_per_shelf if i < cluster_shelves - 1 else len(cluster_skus)
                
                shelf_allocation[current_shelf].extend(cluster_skus[start_idx:end_idx])
                current_shelf += 1
        
        # 处理噪声点
        if -1 in self.clusters:
            noise_skus = self.clusters[-1]['skus']
            self._distribute_remaining_skus(noise_skus, shelf_allocation, num_shelves)
        
        return shelf_allocation
    
    def _sales_based_allocation(self, num_shelves):
        """
        基于销量的分配策略
        """
        shelf_allocation = {i: [] for i in range(num_shelves)}
        
        # 按销量排序SKU
        sku_sales = self.analysis_results.get('sku_sales', {})
        sorted_skus = sorted(self.skus, key=lambda x: sku_sales.get(x, 0), reverse=True)
        
        # 平均分配到货架
        skus_per_shelf = len(sorted_skus) // num_shelves
        
        for i in range(num_shelves):
            start_idx = i * skus_per_shelf
            end_idx = start_idx + skus_per_shelf if i < num_shelves - 1 else len(sorted_skus)
            shelf_allocation[i] = sorted_skus[start_idx:end_idx]
        
        return shelf_allocation
    
    def _balanced_allocation(self, num_shelves):
        """
        平衡分配策略
        """
        shelf_allocation = {i: [] for i in range(num_shelves)}
        
        # 简单的轮询分配
        for i, sku in enumerate(self.skus):
            shelf_id = i % num_shelves
            shelf_allocation[shelf_id].append(sku)
        
        return shelf_allocation
    
    def _distribute_remaining_skus(self, remaining_skus, shelf_allocation, num_shelves):
        """
        分配剩余的SKU到货架
        """
        # 找到最少SKU的货架
        for sku in remaining_skus:
            min_shelf = min(shelf_allocation.keys(), 
                          key=lambda x: len(shelf_allocation[x]))
            shelf_allocation[min_shelf].append(sku)
    
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
    
    def visualize_clustering_results(self, save_path="d:\\Mycode\\KIVA\\result\\problem1\\clustering_results.png"):
        """
        可视化聚类结果
        """
        if self.cluster_labels is None:
            print("请先执行聚类分析！")
            return
        
        print("正在生成聚类结果可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 聚类分布饼图
        cluster_counts = Counter(self.cluster_labels)
        valid_clusters = {k: v for k, v in cluster_counts.items() if k != -1}
        noise_count = cluster_counts.get(-1, 0)
        
        if valid_clusters:
            axes[0, 0].pie(valid_clusters.values(), 
                          labels=[f'聚类 {k}' for k in valid_clusters.keys()],
                          autopct='%1.1f%%')
            axes[0, 0].set_title(f'聚类分布 (噪声点: {noise_count}个)')
        
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
    
    def run_complete_allocation_analysis(self, eps=0.5, min_samples=5, num_shelves=20, 
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
        print("开始完整的货架分配分析流程...")
        
        # 1. 参数优化（可选）
        if optimize_params:
            best_params, _ = self.optimize_clustering_parameters()
            if best_params:
                eps = best_params['eps']
                min_samples = best_params['min_samples']
                print(f"使用优化后的参数: eps={eps:.3f}, min_samples={min_samples}")
        
        # 2. 执行聚类
        self.perform_dbscan_clustering(eps=eps, min_samples=min_samples)
        
        # 3. 货架分配
        self.allocate_skus_to_shelves(num_shelves=20, 
                                    allocation_method=allocation_method)
        
        # # 4. 生成可视化
        # self.visualize_clustering_results()
        # self.visualize_shelf_allocation()
        
        # 5. 生成报告
        self.generate_allocation_report()
        
        print("完整的货架分配分析流程执行完成！")
        
        return {
            'clusters': self.clusters,
            'shelf_allocation': self.shelf_allocation,
            'allocation_stats': getattr(self, 'allocation_stats', None)
        }

# 使用示例
if __name__ == "__main__":
    # 这里需要先运行网络分析
    from model1_network_analysis_enhanced import SKUAssociationAnalyzer
    
    # 创建网络分析器
    analyzer = SKUAssociationAnalyzer("d:\\Mycode\\KIVA\\data\\outbound.xlsx")
    
    # 运行网络分析
    if analyzer.run_complete_analysis(threshold=5):
        # 创建货架分配策略
        shelf_strategy = ShelfAllocationStrategy(analyzer)
        
        # 运行完整的分配分析
        results = shelf_strategy.run_complete_allocation_analysis(
            eps=0.8,
            min_samples=3,
            num_shelves=20,
            allocation_method='cluster_based',
            optimize_params=True #启动参数优化
        )
        
        print("货架分配策略分析完成！")
    else:
        print("网络分析失败，无法进行货架分配分析")
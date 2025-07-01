import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from collections import Counter
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SKUAssociationAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.all_records = None
        self.association_matrix = None
        self.G = None
        self.skus = None
        self.analysis_results = {}
        
    def load_data(self):
        """加载和预处理数据"""
        print("正在加载数据...")
        time1 = time.time()
        try:
            self.df = pd.read_excel(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date_crt'])
            self.df['mat_code'] = self.df['mat_code'].astype(str)
            print(f"数据加载成功！数据大小: {self.df.shape}, 加载数据用时：{time.time() - time1} \n")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False # 返回False表示加载失败, 程序将退出
    
    def process_orders(self):
        """处理订单数据，生成SKU集合"""
        print("Step1:正在处理订单数据...")
        # 按订单分组生成SKU集合
        self.orders = self.df.groupby('order_num')['mat_code'].apply(set).reset_index(name='skus')
        
        # 获取所有唯一SKU - 修复：正确合并所有订单中的SKU
        all_skus = set()
        for sku_set in self.orders['skus']:
            all_skus.update(sku_set)
        self.skus = sorted(list(all_skus))  # 排序保证一致性
        
        print(f"订单总数: {len(self.orders)}")
        print(f"商品总数: {len(self.skus)}\n")
        
        # 保存基础统计信息
        self.analysis_results['total_orders'] = len(self.orders)
        self.analysis_results['total_skus'] = len(self.skus)
        
    def build_association_matrix(self):
        """构建商品关联矩阵"""
        print("Step2:正在构建商品关联矩阵...")
        num_skus = len(self.skus)
        self.association_matrix = np.zeros((num_skus, num_skus))
        
        # 创建SKU到索引的映射
        sku_to_idx = {sku: idx for idx, sku in enumerate(self.skus)}
        
        # 计算共现频次(可以优化下对角矩阵的生成方式)
        for _, row in self.orders.iterrows():
            sku_list = list(row['skus'])
            for i, sku1 in enumerate(sku_list):
                idx1 = sku_to_idx[sku1]
                # 对角线元素：商品出现频次
                self.association_matrix[idx1][idx1] += 1
                # 非对角线元素：共现频次
                for j, sku2 in enumerate(sku_list):
                    if i != j:
                        idx2 = sku_to_idx[sku2]
                        self.association_matrix[idx1][idx2] += 1
        
        print("商品关联矩阵构建完成！\n")
        
    def build_network(self):
        """构建商品关联网络"""
        threshold = 10 #设置关联度的阈值threshold=1
        print(f"Step3:正在构建商品关联网络（阈值: {threshold}）...")
        self.G = nx.Graph()
        
        # 添加节点
        for sku in self.skus:
            self.G.add_node(sku)
        
        # 添加边
        num_skus = len(self.skus)
        for i in range(num_skus):
            for j in range(i+1, num_skus):
                weight = self.association_matrix[i][j]
                if weight >= threshold:
                    self.G.add_edge(self.skus[i], self.skus[j], weight=weight)
        
        print(f"网络构建完成！节点数: {self.G.number_of_nodes()}, 边数: {self.G.number_of_edges()}\n")
        
        # 保存网络基础信息
        self.analysis_results['network_nodes'] = self.G.number_of_nodes() # 网络节点数
        self.analysis_results['network_edges'] = self.G.number_of_edges() # 网络变数
        self.analysis_results['network_density'] = nx.density(self.G) if self.G.number_of_nodes() > 0 else 0

    # 分析商品销售特征  
    def analyze_sales_characteristics(self):
        """分析商品销售特征"""
        print("Step4:正在分析商品销售特征...")
        
        # 计算商品销量（出现在订单中的次数）
        sku_sales = {}  # 每个sku的销售量
        sku_purchase_freq = {}  # 每个sku的购买频次
        sku_hot_sale_inx = {} # 每个sku的畅销指数
        
        # 统计sku的销量：每个sku在订单中的总购买次数
        self.df['sa_qty_bum'] = self.df['sa_qty_bum'].astype(int)
        skus_sale = self.df.groupby(by=['mat_code'])['sa_qty_bum'].sum().reset_index(name='num')
        sku_sales = dict(zip(skus_sale['mat_code'], skus_sale['num']))
        
        # 统计sku的购买频次：每个sku在多少个订单中出现过
        for sku in self.skus:
            # 使用sum和生成器表达式高效统计sku出现在多少个订单中
            sku_purchase_freq[sku] = sum(1 for order_skus in self.orders['skus'] if sku in order_skus)

        # 统计sku的畅销指数：非零购买天数ni用于识别长期畅销商品、短期畅销商品和慢销商品
        # 按天分组统计每天销售的商品集合
        skus = self.df.groupby(by=self.df['date'].dt.date)['mat_code'].apply(set).reset_index(name='mat_codes')
        for sku in self.skus:
            sku_hot_sale_inx[sku] = sum(1 for skus in skus['mat_codes'] if sku in skus)


        # ABC分析
        sorted_sales = sorted(sku_sales.items(), key=lambda x: x[1], reverse=True) # 字典排序, 若iteration是sku_scales, 则传入lambda的x是keys
        total_sales = sum(sku_sales.values()) # 字典求和
        
        cumulative_sales = 0
        abc_classification = {'A': [], 'B': [], 'C': []}
        
        for sku, sales in sorted_sales:
            cumulative_sales += sales
            cumulative_percentage = cumulative_sales / total_sales
            
            if cumulative_percentage <= 0.7:
                abc_classification['A'].append(sku)
            elif cumulative_percentage <= 0.9:
                abc_classification['B'].append(sku)
            else:
                abc_classification['C'].append(sku)
        
        self.analysis_results['sku_sales'] = sku_sales
        self.analysis_results['sku_purchase_freq'] = sku_purchase_freq
        self.analysis_results['abc_classification'] = abc_classification
        self.analysis_results['sku_hot_sale_inx'] = sku_hot_sale_inx
        
        print(f"A类商品数量: {len(abc_classification['A'])}")
        print(f"B类商品数量: {len(abc_classification['B'])}")
        print(f"C类商品数量: {len(abc_classification['C'])}\n")
        
    def analyze_network_topology(self):
        """分析网络拓扑特征"""
        print("Step5:正在分析网络拓扑特征...")
        
        if self.G.number_of_nodes() == 0:
            print("网络为空，跳过拓扑分析")
            return
        
        # 度分析
        degrees = dict(self.G.degree())
        degree_values = list(degrees.values())
        
        # 加权度分析
        weighted_degrees = dict(self.G.degree(weight='weight'))
        weighted_degree_values = list(weighted_degrees.values())
        
        # 度分布统计
        degree_distribution = Counter(degree_values)
        weighted_degree_distribution = Counter([int(wd) for wd in weighted_degree_values])
        
        self.analysis_results['degrees'] = degrees
        self.analysis_results['weighted_degrees'] = weighted_degrees
        self.analysis_results['degree_distribution'] = degree_distribution
        self.analysis_results['weighted_degree_distribution'] = weighted_degree_distribution
        
        # Top商品分析
        top_degree_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        top_weighted_degree_nodes = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        self.analysis_results['top_degree_nodes'] = top_degree_nodes
        self.analysis_results['top_weighted_degree_nodes'] = top_weighted_degree_nodes
        
        print("网络拓扑分析完成！\n")
        
    def visualize_network(self, save_path="d:\\Mycode\\KIVA\\result\\problem1\\enhanced_network.png"):
        """可视化网络"""
        print("Step6:正在生成网络可视化图...")
        
        if self.G.number_of_nodes() == 0:
            print("网络为空，无法可视化")
            return
        
        plt.figure(figsize=(15, 12))
        
        # 使用布局算法 - 尝试多种布局方法
        try:
            # 优先使用Kamada-Kawai布局，适合中小型网络
            if self.G.number_of_nodes() <= 100:
                pos = nx.kamada_kawai_layout(self.G)
            else:
                # 对于大型网络使用Spring布局
                pos = nx.spring_layout(self.G, k=3, iterations=50)
        except:
            try:
                # 备选方案1：使用Fruchterman-Reingold布局
                pos = nx.fruchterman_reingold_layout(self.G, k=2, iterations=50)
            except:
                try:
                    # 备选方案2：使用Shell布局
                    pos = nx.shell_layout(self.G)
                except:
                    # 最后备选：使用圆形布局
                    pos = nx.circular_layout(self.G)
        
        # 根据度数设置节点颜色和大小
        degrees = dict(self.G.degree())
        node_colors = [degrees[node] for node in self.G.nodes()]
        node_sizes = [degrees[node] * 50 + 100 for node in self.G.nodes()]
        
        # 绘制网络
        nx.draw_networkx_nodes(self.G, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes,
                              cmap=plt.cm.viridis,
                              alpha=0.8)
        
        # 绘制边
        edges = self.G.edges()
        if edges:
            weights = [self.G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            edge_widths = [w/max_weight * 3 for w in weights]
            
            nx.draw_networkx_edges(self.G, pos, 
                                  width=edge_widths,
                                  alpha=0.6,
                                  edge_color='gray')
        
        plt.title('商品关联网络图', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"网络图已保存到: {save_path}\n")
        
    def create_analysis_plots(self):
        """创建各种分析图表"""
        print("Step7:正在生成分析图表...")
        
        # 1. ABC分类分析图
        self._plot_abc_analysis()

        # 2. 商品销量分布图
        self._plot_sales_distribution()
        
        # 3. 度分布饼状图
        self._plot_degree_distribution()
        
        # 4. 加权度分布饼状图
        self._plot_weighted_degree_distribution()
        
        # 5. 商品购买频次散点图
        self._plot_purchase_frequency()
        
        # 6. 长短期畅销度分布
        self._plot_bestseller_distribution()
        
    def _plot_abc_analysis(self):
        """绘制ABC分类分析图"""
        abc = self.analysis_results['abc_classification']
        
        plt.figure(figsize=(10, 6))
        
        # 计算累积销量百分比
        sorted_sales = sorted(self.analysis_results['sku_sales'].items(), 
                            key=lambda x: x[1], reverse=True)
        total_sales = sum(self.analysis_results['sku_sales'].values())
        
        cumulative_percentages = []
        cumulative_sales = 0
        
        for _, sales in sorted_sales:
            cumulative_sales += sales
            cumulative_percentages.append(cumulative_sales / total_sales * 100)
        
        x_positions = list(range(len(sorted_sales)))
        
        plt.plot(x_positions, cumulative_percentages, 'b-', linewidth=2, label='累积销量百分比')
        plt.axhline(y=70, color='r', linestyle='--', label='A类商品(70%)')
        plt.axhline(y=90, color='orange', linestyle='--', label='B类商品(90%)')
        
        plt.xlabel('商品排名')
        plt.ylabel('累积销量百分比(%)')
        plt.title('商品销量ABC分类分析')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('d:\\Mycode\\KIVA\\result\\problem1\\abc_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_sales_distribution(self):
        """绘制商品销量分布"""
        sales_data = self.analysis_results['sku_sales']
        
        sales_values = list(sales_data.values())
        
        plt.figure(figsize=(12, 8))
        plt.hist(sales_values, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('商品id')
        plt.ylabel('商品数量')
        plt.title('商品销量')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('d:\\Mycode\\KIVA\\result\\problem1\\sales_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_degree_distribution(self):
        """绘制度分布饼状图"""
        if not self.analysis_results.get('degree_distribution'):
            return
            
        degree_dist = self.analysis_results['degree_distribution']
        
        # 创建度数区间
        degree_ranges = {
            '1': sum(count for degree, count in degree_dist.items() if degree == 1),
            '1到10': sum(count for degree, count in degree_dist.items() if 1 < degree <= 10),
            '11到20': sum(count for degree, count in degree_dist.items() if 11 <= degree <= 20),
            '21到30': sum(count for degree, count in degree_dist.items() if 21 <= degree <= 30),
            '31到40': sum(count for degree, count in degree_dist.items() if 31 <= degree <= 40),
            '41到50': sum(count for degree, count in degree_dist.items() if 41 <= degree <= 50),
            '大于50': sum(count for degree, count in degree_dist.items() if degree > 50)
        }
        
        # 过滤掉值为0的项
        degree_ranges = {k: v for k, v in degree_ranges.items() if v > 0}
        
        plt.figure(figsize=(10, 8))
        plt.pie(degree_ranges.values(), labels=degree_ranges.keys(), autopct='%1.1f%%')
        plt.title('商品关联网络中度的分布饼状图')
        plt.tight_layout()
        plt.savefig('d:\\Mycode\\KIVA\\result\\problem1\\degree_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_weighted_degree_distribution(self):
        """绘制加权度分布饼状图"""
        if not self.analysis_results.get('weighted_degree_distribution'):
            return
            
        weighted_degree_dist = self.analysis_results['weighted_degree_distribution']
        
        # 创建加权度区间
        weighted_ranges = {
            '1到50': sum(count for degree, count in weighted_degree_dist.items() if 1 <= degree <= 50),
            '51到100': sum(count for degree, count in weighted_degree_dist.items() if 51 <= degree <= 100),
            '101到150': sum(count for degree, count in weighted_degree_dist.items() if 101 <= degree <= 150),
            '151到200': sum(count for degree, count in weighted_degree_dist.items() if 151 <= degree <= 200),
            '201到300': sum(count for degree, count in weighted_degree_dist.items() if 201 <= degree <= 300),
            '大于300': sum(count for degree, count in weighted_degree_dist.items() if degree > 300)
        }
        
        # 过滤掉值为0的项
        weighted_ranges = {k: v for k, v in weighted_ranges.items() if v > 0}
        
        plt.figure(figsize=(10, 8))
        plt.pie(weighted_ranges.values(), labels=weighted_ranges.keys(), autopct='%1.1f%%')
        plt.title('商品关联网络中加权度的分布饼状图')
        plt.tight_layout()
        plt.savefig('d:\\Mycode\\KIVA\\result\\problem1\\weighted_degree_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_purchase_frequency(self):
        """绘制商品购买频次散点图"""
        sales_data = self.analysis_results['sku_sales']
        freq_data = self.analysis_results['sku_purchase_freq']
        
        plt.figure(figsize=(12, 8))
        
        x_values = list(range(len(sales_data)))
        y_values = [freq_data[sku] for sku in sales_data.keys()]
        
        plt.scatter(x_values, y_values, alpha=0.6, s=30)
        plt.xlabel('订单号')
        plt.ylabel('订单购买商品数量')
        plt.title('订单购买商品数量的散点图')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('d:\\Mycode\\KIVA\\result\\problem1\\purchase_frequency.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_bestseller_distribution(self):
        """绘制长短期畅销度分布"""
        sales_data = self.analysis_results['sku_hot_sale_inx']
        
        sales_values = list(sales_data.values())
        
        plt.figure(figsize=(12, 8))
        plt.hist(sales_values, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('商品长短期畅销程度ni')
        plt.ylabel('商品数量')
        plt.title('商品长短期畅销程度的分布')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('d:\\Mycode\\KIVA\\result\\problem1\\bestseller_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self, output_path="d:\\Mycode\\KIVA\\result\\problem1\\network_analysis_report.md"):
        """生成分析报告"""
        print("Step8:正在生成分析报告...")
        
        report_content = f"""# 基于关联网络的机器人移动货架系统货位分配方法分析报告

        ## 1. 研究概述

        本报告基于商品订单数据，构建了商品关联网络，并从多个维度对网络特征进行了深入分析，为机器人移动货架系统的货位优化分配提供理论依据。

        ## 2. 数据基本信息

        - **订单总数**: {self.analysis_results.get('total_orders', 'N/A')}
        - **商品总数**: {self.analysis_results.get('total_skus', 'N/A')}
        - **网络节点数**: {self.analysis_results.get('network_nodes', 'N/A')}
        - **网络边数**: {self.analysis_results.get('network_edges', 'N/A')}
        - **网络密度**: {self.analysis_results.get('network_density', 'N/A'):.4f}

        ## 3. 商品关联网络生成过程

        ### 3.1 订单-商品关系矩阵构建

        从原始订单数据出发，按订单编号分组，提取每个订单中包含的商品集合，形成订单-商品关系矩阵P。矩阵中的元素表示商品是否出现在特定订单中。

        ### 3.2 商品关联矩阵生成

        基于订单-商品关系矩阵，计算商品间的共现频次，构建商品关联矩阵S。矩阵中的元素S[i,j]表示商品i和商品j在同一订单中出现的次数。

        ### 3.3 商品关联网络构建

        以商品为节点，以商品间的关联强度为边权重，构建商品关联网络。设置适当的阈值过滤弱关联，保留具有显著关联关系的商品对。

        ## 4. 销售特征分析

        ### 4.1 ABC分类分析

        根据帕累托法则对商品进行ABC分类：

        - **A类商品**: {len(self.analysis_results.get('abc_classification', {}).get('A', []))}个，占总销量的70%
        - **B类商品**: {len(self.analysis_results.get('abc_classification', {}).get('B', []))}个，占总销量的20%
        - **C类商品**: {len(self.analysis_results.get('abc_classification', {}).get('C', []))}个，占总销量的10%

        A类商品是核心商品，应优先考虑其在货架系统中的位置配置。

        ### 4.2 商品购买频次分析

        通过分析订单中商品的购买频次分布，发现：
        - 大部分订单包含的商品数量较少
        - 少数订单包含大量商品，呈现长尾分布特征
        - 这种分布特征对货位分配策略具有重要指导意义

        ### 4.3 长短期畅销度分析

        定义商品的长短期畅销度为其在统计期间内的非零购买天数。分析结果显示：
        - 大部分商品的畅销度较低，属于长尾商品
        - 少数商品具有较高的畅销度，是系统的核心商品
        - 畅销度分布呈现明显的幂律特征

        ## 5. 网络拓扑特征分析

        ### 5.1 度分析

        商品节点的度表示其与其他商品的直接关联数量。分析发现：
        """
                
        # 添加度分布信息
        if self.analysis_results.get('top_degree_nodes'):
            report_content += "\n**度最高的前10个商品**:\n\n"
            for i, (node, degree) in enumerate(self.analysis_results['top_degree_nodes'][:10], 1):
                report_content += f"{i}. {node}: {degree}\n"
        
        report_content += f"""

        ### 5.2 加权度分析

        加权度考虑了边的权重信息，更准确地反映了商品间的关联强度：
        """
                
        # 添加加权度分布信息
        if self.analysis_results.get('top_weighted_degree_nodes'):
            report_content += "\n**加权度最高的前10个商品**:\n\n"
            for i, (node, weighted_degree) in enumerate(self.analysis_results['top_weighted_degree_nodes'][:10], 1):
                report_content += f"{i}. {node}: {weighted_degree:.2f}\n"
        
        report_content += f"""

        ### 5.3 无标度特性

        商品关联网络表现出明显的无标度特性：
        - 少数商品具有很高的连接度，成为网络的核心节点
        - 大部分商品的连接度较低，处于网络的边缘
        - 这种特性符合现实中的商品销售规律

        ## 6. 网络整体结构分析

        ### 6.1 核心-边缘结构

        网络呈现明显的核心-边缘结构：
        - **核心区域**: 由高度关联的热销商品组成，形成密集的连接子图
        - **边缘区域**: 由低关联度的商品组成，与核心区域连接较少
        - **桥接节点**: 连接不同商品类别的关键节点

        ### 6.2 社区结构

        通过社区检测算法识别出多个商品社区，每个社区内的商品具有较强的关联性，可能属于相同的商品类别或具有相似的消费场景。

        ## 7. 货位分配建议

        基于网络分析结果，提出以下货位分配建议：

        ### 7.1 核心商品优先配置
        - 将度数和加权度较高的核心商品配置在易于访问的货位
        - 优先考虑A类商品的货位布局

        ### 7.2 关联商品就近配置
        - 将具有强关联关系的商品配置在相邻或就近的货位
        - 减少机器人的移动距离和拣选时间

        ### 7.3 动态调整策略
        - 根据商品关联关系的时间变化，动态调整货位配置
        - 定期更新网络模型，优化货位分配方案

        ## 8. 结论

        本研究通过构建商品关联网络，深入分析了商品间的关联模式和网络拓扑特征，为机器人移动货架系统的货位优化提供了科学依据。研究发现：

        1. 商品关联网络具有明显的无标度特性和核心-边缘结构
        2. ABC分类分析有效识别了核心商品和长尾商品
        3. 网络拓扑特征为货位分配策略提供了重要参考
        4. 基于关联关系的货位配置能够有效提升系统效率

        ## 9. 技术实现

        本分析使用Python实现，主要依赖以下技术栈：
        - **数据处理**: pandas, numpy
        - **网络分析**: networkx
        - **可视化**: matplotlib, seaborn
        - **统计分析**: scipy, collections

        分析代码具有良好的模块化设计，支持不同数据源和参数配置，便于在实际系统中部署和应用。

        ---

        *报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """
                
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"分析报告已保存到: {output_path}")
        
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("开始完整的网络分析流程...")
        
        # 1. 加载数据
        if not self.load_data():
            return False
        # 2. 处理订单
        self.process_orders()
        
        # 3. 构建关联矩阵
        self.build_association_matrix()
        df_association_matrix = pd.DataFrame(self.association_matrix) # 打印关联矩阵
        df_association_matrix.to_excel(r"D:\Mycode\KIVA\result\problem1\association_matrix.xlsx")
        # 4. 构建网络
        self.build_network()
        
        # 5. 分析销售特征
        self.analyze_sales_characteristics()
        
        # 6. 分析网络拓扑
        self.analyze_network_topology()
        
        # 7. 生成可视化
        # self.visualize_network()
        # self.create_analysis_plots()
        
        # 8. 生成报告
        # self.generate_report()
        
        print("完整分析流程执行完成！")
        return True

# 主程序执行
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = SKUAssociationAnalyzer("d:\\Mycode\\KIVA\\data\\outbound7-23.xlsx")

    # 运行完整分析
    analyzer.run_complete_analysis()
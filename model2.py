import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import sys
from read_data import Read_data

def Gurobi_soluton(N, M, P, K, E, t_0, a, x):

    # 模型构建
    model = gp.Model("OrderBatchScheduling")

    # 定义决策变量
    z = model.addVars(N, K, vtype=GRB.BINARY, name="z")  # 订单分配
    w = model.addVars(K, M, vtype=GRB.BINARY, name="w")  # 货架搬运
    u = model.addVars(P, K, vtype=GRB.BINARY, name="u")  # 商品批次归属
    max_C = model.addVar(vtype=GRB.CONTINUOUS, name="max_C")
    model.update()

    # 目标函数：最小化最大完工时间
    model.setObjective(max_C, GRB.MINIMIZE)

    # 约束条件
    # 约束10：每个订单必须分配到且仅分配到一个批次
    for n in range(N):
        model.addConstr(gp.quicksum(z[n, k] for k in range(K)) == 1, name=f"C10_{n}")

    # 约束11：每个批次订单数不超过E
    for k in range(K):
        model.addConstr(gp.quicksum(z[n, k] for n in range(N)) <= E, name=f"C11_{k}")

    # 约束12：商品批次归属逻辑
    for i in range(P):
        for k in range(K):
            expr = gp.LinExpr()
            for n in range(N):
                expr += a[i, n] * z[n, k]
            model.addConstr(expr <= P * u[i, k], name=f"C12_{i}_{k}")

    # 约束13：货架必须包含批次所有商品
    for i in range(P):
        for k in range(K):
            model.addConstr(gp.quicksum(x[i, m] * w[k, m] for m in range(M)) >= u[i, k], name=f"C13_{i}_{k}")

    # 约束14：每个批次k的完工时间不超过max_C
    for k in range(K):
        model.addConstr(max_C >= gp.quicksum(w[k, m] * t_0 for m in range(M)), name=f"maxC_batch_{k}")

    # 模型求解与结果输出
    model.setParam('OutputFlag', 1)
    model.setParam('MIPFocus', 3)
    model.setParam('Cuts', 1) # 关闭切割
    model.setParam('Heuristics', 0.95)

    model.optimize()
    # model_relaxed = model.relax()
    # model_relaxed.optimize()

    # 检查求解状态
    # 模型求解成功
    if model.status == GRB.OPTIMAL:
        # pass
        print(f"\n\n----------------------优化结果----------------------")
        print(f"最小完工时间 {model.objVal} min ")
        # 输出批次分配详情
        for k in range(K):
            assigned_orders = [n for n in range(N) if z[n, k].X >= 1]
            if assigned_orders:
                print(f"批次{k + 1}-{len(assigned_orders)}个订单-{assigned_orders}")
                moved_shelves = [m for m in range(M) if w[k, m].X >= 1]
                print(f"搬运货架{len(moved_shelves)}: {moved_shelves}")

    # 其他状态
    elif model.status == GRB.INF_OR_UNBD:
        print('Model is infeasible or unbounded')
        sys.exit(0)
    elif model.status == GRB.INFEASIBLE:
        print('Model is infeasible')
    elif model.status == GRB.UNBOUNDED:
        print('Model is unbounded')
        sys.exit(0)
    else:
        print('Optimization ended with status %d' % model.Status)
        sys.exit(0)

    return model.objVal

# # --------------------------
# # 模型分析工具（可选）
# # --------------------------
# # 1. 松弛模型分析
# model_relaxed = model.relax()
# model_relaxed.optimize()
# print(f"\n松弛模型下界: {model_relaxed.objVal}")
#
# # 2. 灵敏度分析
# for v in model.getVars()[:10]:  # 查看前10个变量
#     print(f"{v.VarName}: Reduced Cost = {v.RC}")

if __name__ == '__main__':

    # 设置参数
    K = 5  # 最大批次数量
    E = 36  # 每批次最大订单数,即分拨墙容量
    t_0 = 1  # 货架分拣时间(min)

    # 实际数据读取
    date_range = "2019-07-23"  # 输入日期哦格式： YYYY-MM-DD
    order_range = 400  # 读取7月23日，前400条数据,124个订单,总订单数不能超过 K * E
    outbound_path = r'KIVA\data\outbound7-23.xlsx'
    inventory_path = r'KIVA\data\inventory7-23.xlsx'
    timer1 = time.time()
    N, a_in, M, x_im, P = Read_data(outbound_path, inventory_path, date_range, order_range)
    timer2 = time.time()
    print(f"读取数据时间{timer2 - timer1}s")
    print(f"批次中的物料数:{a_in.shape[0]} 订单数:{a_in.shape[1]}")

    # 调用Gurobi求解模型
    objVal = Gurobi_soluton(N, M, P, K, E, t_0, a_in, x_im)
    timer3 = time.time()
    print(f"----------------------模型求解用时----------------------")
    print(f"第1次求解\t前{order_range}条订单\t读取数据时间{timer2-timer1}\t目标函数值{objVal}min\t求解用时{timer3-timer2}")


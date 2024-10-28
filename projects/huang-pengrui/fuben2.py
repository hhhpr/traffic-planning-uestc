import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

q = 1000  # 流量更新的基础值

# 生成蓝噪声样本
def generate_blue_noise(num_points, width, height, radius, fixed_positions):
    """
    生成指定数量的蓝噪声样本点，确保点之间的距离不小于半径。

    :param num_points: 生成的点数
    :param width: 区域宽度
    :param height: 区域高度
    :param radius: 最小间距半径
    :param fixed_positions: 固定位置的点
    :return: 生成的点数组
    """
    points = fixed_positions.tolist()  # 将固定位置转换为列表
    while len(points) < num_points:
        point = np.random.rand(2) * [width, height]  # 随机生成点
        # 检查新点与已有点的距离
        if all(np.linalg.norm(point - np.array(p)) >= radius for p in points):
            points.append(point.tolist())  # 将新点添加为列表
    return np.array(points)

# 定义节点及边权重
def initialize_graph_with_edges(points):
    """
    初始化图，节点之间根据距离生成边并赋予权重。

    :param points: 节点坐标数组
    :return: 图结构和节点列表
    """
    nodes = [f'Node {i}' for i in range(len(points))]
    graph = {node: {} for node in nodes}  # 初始化图结构
    for i, point in enumerate(points):
        distances = []
        for j, other_point in enumerate(points):
            if i != j:
                dist = np.linalg.norm(point - other_point)  # 计算距离
                distances.append((dist, nodes[j]))
        
        distances.sort(key=lambda x: x[0])  # 按距离排序
        
        for _, neighbor in distances[:3]:  # 连接最近的三个邻居
            graph[nodes[i]][neighbor] = _  # 添加边权重

    return graph, nodes

# Dijkstra算法
def dijkstra(graph, start, end):
    queue = [(0, start, [])]  # 初始化优先队列
    seen = set()  # 记录已处理的节点
    min_dist = {start: 0}  # 记录最小距离

    while queue:
        (cost, node, path) = heapq.heappop(queue)  # 弹出最小成本节点
        if node in seen:
            continue
        seen.add(node)
        path = path + [node]

        if node == end:
            return (cost, path)  # 找到终点，返回路径和距离

        for neighbor, weight in graph[node].items():
            if neighbor in seen:
                continue
            prev = min_dist.get(neighbor, None)
            next = cost + weight  # 计算当前路径的成本
            if prev is None or next < prev:
                min_dist[neighbor] = next
                heapq.heappush(queue, (next, neighbor, path))  # 更新优先队列

    return float("inf"), []

# 更新流量q
def update_flow(qgraph, path):

    # 所有边的流量降低
    for node in qgraph.keys():
        for neighbor in qgraph[node].keys():
            qgraph[node][neighbor] *= 0.9

    # 当前最短路径流量增加
    for i in range(len(path) - 1):
        node = path[i]  # 当前节点
        next_node = path[i + 1]  # 下一个节点
        if next_node in qgraph[node]:
            qgraph[node][next_node] += q * 0.1

# 更新t
def update_t(graph, qgraph):
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
        
            graph[node][neighbor] *= trans(qgraph[node][neighbor])

# BPR函数
def trans(q):
    return 1 + (q / 1000) ** 2  


# 可视化图形
def visualize_graph(ax, graph, points, qgraph):
    ax.clear()  # 清空轴
    for node, (x, y) in zip(graph.keys(), points):
        ax.scatter(x, y, color='blue')  # 绘制节点
        ax.text(x, y, node, fontsize=12, ha='right')  # 显示节点名称

    drawn_edges = set()  # 存储已经绘制过的边
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            edge = tuple(sorted((node, neighbor)))
            start = points[int(node.split()[1])]
            end = points[int(neighbor.split()[1])]
            flow = qgraph[node][neighbor]
            if flow > 0:  # 流量大于0时绘制边
                linewidth = max(0.1, flow / 100)  # 根据流量调整线宽
                color = 'blue'
            else:  # 流量等于0时绘制橙色边
                linewidth = 0.1
                color = 'orange'
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=linewidth, alpha=0.7)  # 绘制边
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            if edge not in drawn_edges and flow > 0:
                ax.text(mid_x, mid_y, f'{flow:.2f}', fontsize=10, color='red')  # 显示流量具体数值
                drawn_edges.add(edge)  # 标记这条边已经绘制过

    ax.set_xlim(0, 100)  # 设置x轴范围
    ax.set_ylim(0, 100)  # 设置y轴范围

# 主循环
def main():
    width, height = 100, 100
    num_points = 20  # 节点数量
    radius = 15  # 增加半径，确保点间距

    # 固定位置的节点，Node 0 在左边，Node 19 在右边
    fixed_positions = np.array([[5, 25], [95, 95]])
    points = generate_blue_noise(num_points, width, height, radius, fixed_positions)
    graph, nodes = initialize_graph_with_edges(points)  # 初始化图
    distance, path = dijkstra(graph, 'Node 0', 'Node 19')  # 计算最短路径

    # 初始化q图
    qgraph = {node: {neighbor: 0 for neighbor in graph[node]} for node in graph}

    # 初始化q图
    for node in graph:
        for neighbor in graph[node]:
            if node in path and neighbor in path and path.index(neighbor) == path.index(node) + 1:
                qgraph[node][neighbor] = 1000  # 为路径上的边分配初始流量
            else:
                qgraph[node][neighbor] = 0
    
    fig, ax = plt.subplots()  # 创建绘图对象

    def update(frame):
        distance, path = dijkstra(graph, 'Node 0', 'Node 19')  # 重新计算最短路径
        update_flow(qgraph, path)  # 更新q
        update_t(graph, qgraph)  # 更新t
        visualize_graph(ax, graph, points, qgraph)  # 可视化
        ax.set_title(f"Iteration {frame + 1}")  # 更新标题

    ani = FuncAnimation(fig, update, frames=100, repeat=False)
    plt.show()

main()

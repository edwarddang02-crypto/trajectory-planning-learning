# 本例：在有障碍物的二维网格上，比较 Dijkstra 和 A* 的最短路径搜索。
#
# 建议按这个顺序阅读：
# 1. 地图设置：明确起点、终点、障碍物，以及允许怎样移动。
# 2. neighbors / manhattan：分别回答“下一步能去哪”和“离终点估计多远”。
# 3. search：不断取出最值得探索的格子，更新到各格子的最小已知代价。
# 4. validate：检查搜索结果是否合法。
# 5. 文件底部的 for：程序实际从这里调用搜索，分别运行两种算法并打印地图。
#
# 核心区别：两种算法共用 search，只是传入的 heuristic（启发函数）不同。
#   g(n)：从起点走到 n 的当前已知代价，也就是代码中的 cost。
#   h(n)：从 n 到终点的估计剩余代价，也就是 heuristic(n)。
#   f(n) = g(n) + h(n)：决定哪个格子优先被取出来探索。
# Dijkstra 使用 h(n)=0，优先探索当前走过路程最短的格子。
# A* 使用曼哈顿距离作为 h(n)，在考虑已走路程的同时，向终点方向引导搜索。
# 本例每步代价都为 1，因此最小代价就是最少步数；这种地图也可以用 BFS 求解。

# heapq 提供最小堆：heappush 放入候选项，heappop 取出优先级最小的候选项。
# 注意：堆并不保证整个列表完全有序，只保证能够高效地取出最小项。
from heapq import heappop, heappush
# count() 生成 0、1、2……，用于给每次入队操作分配唯一序号。
from itertools import count
# inf 表示正无穷：尚未到达的格子，其已知代价可先看作无穷大。
from math import inf

# Matplotlib 负责绘图；FuncAnimation 按固定时间间隔刷新画面。
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# 坐标用 (x, y) 表示。x 的范围是 0~9，y 的范围是 0~7。
# 打印时 y 从上往下增加，x 从左往右增加。
WIDTH, HEIGHT = 10, 8
# 集合推导式：把 x=4 这一列设成墙，但在 y=6 处留一个缺口。
# 集合中保存的是不能进入的格子坐标，例如 (4, 0)、(4, 1)。
BLOCKED = {(4, y) for y in range(HEIGHT) if y != 6}
# 起点和终点都在 y=1 这一行，但中间隔着墙，必须绕到 y=6 的缺口。
# 最短步数：从 y=1 下行 5 步 + 从 x=1 右行 7 步 + 上行 5 步 = 17。
START, GOAL = (1, 1), (8, 1)


def neighbors(node):
    # 将当前格子的坐标元组拆成 x 和 y。
    x, y = node
    # 按右、下、左、上的固定顺序尝试移动；不允许斜着走。
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        nxt = (x + dx, y + dy)
        # 同时满足“没有越界”和“不是障碍物”，才是合法的相邻格子。
        if (0 <= nxt[0] < WIDTH and 0 <= nxt[1] < HEIGHT
                and nxt not in BLOCKED):
            # yield 使这个函数成为生成器：每次交出一个合法邻居。
            # search 中的 for 会逐个拿到这些邻居，无须先建立完整列表。
            yield nxt


def manhattan(node):
    # 曼哈顿距离 = 横向坐标差的绝对值 + 纵向坐标差的绝对值。
    # 例如起点 (1, 1) 到终点 (8, 1) 的估计值是 7。
    # 它忽略墙，所以估计 7 步，而实际需要 17 步。
    # 对本例的四方向、每步代价为 1 的地图，它不会高估真正的最短剩余路程。
    # 它也满足一致性：走一步后，估计距离的变化不超过这一步的代价。
    # 因而采用它的 A* 能在终点作为有效候选出堆时，确定最短路已经找到。
    return abs(node[0] - GOAL[0]) + abs(node[1] - GOAL[1])


def search(heuristic, trace=None):
    # trace 是用来保存动画快照的列表；默认 None 表示不记录。
    # 搜索仍然先完整执行，之后再回放快照，不在搜索循环里等待绘图。
    explored = set()
    # heuristic 是传入的函数，可通过 heuristic(某个坐标) 得到 h 值。
    # 本例两次调用分别传入“始终返回 0 的函数”和 manhattan 函数。
    serial = count()
    # queue 是待探索候选项组成的最小堆，不是先进先出的普通队列。
    # 每个候选项是四元组：(f 值, 入队序号, g 值, 格子坐标)。
    # Python 比较元组时从左到右比较：先比 f，f 相同再比序号。
    # 因此 f 相同的候选项按入队先后处理，后面的 g 和坐标不会参与打破平局。
    # 起点的 g=0，所以起点 f=0+h(START)。这里只放一个元素，天然就是堆。
    queue = [(heuristic(START), next(serial), 0, START)]
    # best[格子]：目前发现的、从起点到该格子的最小 g 值。
    # “目前发现”不代表一写入就已确定最优，后面仍可能被更短路线更新。
    best = {START: 0}
    # parent[格子]：到达该格子的当前最佳路线中，上一个格子是谁。
    # 起点没有上一个格子，用 None 表示；找到终点后据此倒推完整路径。
    parent = {START: None}
    # 统计实际展开的次数：有效出堆后，继续检查其邻居才算展开。
    # 过期候选项和终点不计入本例的 expanded。
    expanded = 0
    while queue:
        # 取出当前 f 最小的候选项。“_”表示此处不需要使用该字段的值。
        # 出堆后用保存的 g 和坐标继续处理，无须重新使用 f 或序号。
        _, _, cost, node = heappop(queue)
        # 同一个格子可能先通过较长路线入堆，后来又通过更短路线再次入堆。
        # 更新 best 时没有删除堆中的旧项，因此这里跳过已经过期的旧候选。
        # 例如旧项保存 cost=8，但 best[node] 已更新为 6，就不用再处理旧项。
        if cost != best[node]:
            continue
        # 在“有效出堆之后、判断终点之前”拍一张快照。
        # explored.copy() 固定此刻的状态，避免后续修改集合改变旧画面。
        # frontier 只保留堆中尚未过期的候选坐标；重复坐标合并为一个格子。
        # 因而每帧表示：橙色节点刚出堆，还没有检查它的邻居。
        if trace is not None:
            frontier = {p for _, _, g, p in queue if g == best[p]}
            trace.append((node, cost, expanded, explored.copy(), frontier))
        # 必须等终点作为有效候选被取出，才返回结果。
        # 仅仅在邻居中首次发现终点，通常还不足以证明找到了最短路径。
        if node == GOAL:
            path = []
            # 依次记录终点、终点的父节点……直到起点的父节点 None。
            # 这里改动的是局部变量 node，不会修改 START、GOAL 或地图。
            while node is not None:
                path.append(node)
                node = parent[node]
            # 列表切片 [::-1] 将倒序路径翻转，得到“起点 -> 终点”。
            # cost 仍是刚才终点出堆时的 g 值，即整条路径的步数。
            return path[::-1], cost, expanded
        expanded += 1
        explored.add(node)
        for nxt in neighbors(node):
            # 从当前格子再走一步，得到经过 node 到达 nxt 的候选代价。
            candidate = cost + 1
            # 字典 get 的第二个参数是找不到键时的默认值。
            # 首次发现 nxt 时，有限代价一定小于 inf，所以会进入更新分支。
            # 已发现 nxt 时，只有严格更短才更新；相同代价保留原来的父节点。
            if candidate < best.get(nxt, inf):
                # 这一步称为“松弛”：发现更短路线后，更新代价和前驱。
                best[nxt] = candidate
                parent[nxt] = node
                # 将新的候选项入堆：f = 新的 g + 估计剩余距离 h。
                # 入堆不等于马上探索，要等它成为当前优先级最小的候选项。
                heappush(queue, (candidate + heuristic(nxt),
                                 next(serial), candidate, nxt))
    # 待探索候选全部耗尽仍没到终点，说明无路可走。
    # 空列表表示没有路径，inf 表示不可达；本例的固定地图不会走到这里。
    return [], inf, expanded


def validate(path, cost):
    # assert 是自检：条件为假时抛出 AssertionError，停止程序。
    # 本例调用前已检查 path 非空，因此这里可以安全访问第一个和最后一个元素。
    # 使用 python -O 运行时 assert 会被禁用，所以应以普通模式运行本演示。
    assert path[0] == START and path[-1] == GOAL
    # 路径记录的是格子，代价记录的是移动次数：经过 N 个格子只需走 N-1 步。
    assert cost == len(path) - 1
    # 检查路径上每一个格子：必须在地图内，而且不能穿墙。
    for x, y in path:
        assert 0 <= x < WIDTH and 0 <= y < HEIGHT
        assert (x, y) not in BLOCKED
    # zip(path, path[1:]) 配对相邻节点：(第1个, 第2个)、(第2个, 第3个)……
    # 坐标差绝对值之和必须恰好为 1，防止跳格、斜走或原地停留。
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


# 将搜索快照变成动画。results 中每项保存一种算法的名称、启发函数、路径和快照。
def visualize(results):
    # 使用 Windows 常见中文字体。格子数组的索引是 [y][x]，坐标仍写作 (x, y)。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 数字 0~5 分别映射为：空地、墙、已展开、待探索、当前节点、最终路径。
    colors = ["#ffffff", "#303642", "#9bc9ee", "#ffe293", "#ff8b3d", "#56bf92"]
    labels = ["空地", "障碍", "已展开", "待探索", "当前出堆", "最终路径"]
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    images = []
    for ax in axes:
        image = ax.imshow([[0] * WIDTH for _ in range(HEIGHT)],
                          cmap=cmap, vmin=0, vmax=5, origin="upper")
        images.append(image)
        ax.set_xticks(range(WIDTH))
        ax.set_yticks(range(HEIGHT))
        # 网格线画在相邻整数坐标的中间，整数坐标对应每个格子的中心。
        ax.set_xticks([x - 0.5 for x in range(WIDTH + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(HEIGHT + 1)], minor=True)
        ax.grid(which="minor", color="#c8cdd4", linewidth=0.7)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel("x")
        ax.set_ylabel("y（向下增加）")
        ax.text(*START, "S", ha="center", va="center", weight="bold")
        ax.text(*GOAL, "G", ha="center", va="center", weight="bold")
    fig.legend(handles=[Patch(facecolor=c, label=t) for c, t in zip(colors, labels)],
               loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.09, 1, 0.90))

    def update(frame):
        # 两种算法每帧各前进一步，先完成的算法停留在最终结果。
        for ax, image, (name, heuristic, path, trace) in zip(axes, images, results):
            node, g, expanded, explored, frontier = trace[min(frame, len(trace) - 1)]
            cells = [[0] * WIDTH for _ in range(HEIGHT)]
            for points, value in ((BLOCKED, 1), (explored, 2), (frontier, 3)):
                for x, y in points:
                    cells[y][x] = value
            done = node == GOAL
            if done:
                for x, y in path:
                    cells[y][x] = 5
            else:
                cells[node[1]][node[0]] = 4
            image.set_data(cells)
            h = heuristic(node)
            status = "完成" if done else "展开"
            ax.set_title(f"{name} | expanded={expanded}\n"
                         f"node={node}, g={g}, h={h}, f={g + h}\n{status}")

    # interval 单位是毫秒：300 表示每 0.3 秒前进一步；调大可放慢动画。
    # 必须保留 animation 变量，防止动画对象在窗口播放前被回收。
    animation = FuncAnimation(fig, update, frames=max(len(r[3]) for r in results),
                              interval=1000, repeat=False)
    plt.show()
    return animation


# 上面的 def 只定义函数；下面开始依次调用它们，真正执行搜索和打印。
# 这个循环位于文件顶层，因此直接运行文件或首次导入此模块时都会执行。
# 每轮把一组数据拆成 name 和 heuristic：
#   第一轮：name="Dijkstra"，heuristic=lambda node: 0。
#   第二轮：name="A*"，heuristic=manhattan。
# lambda node: 0 是匿名函数，接受坐标参数，但无论输入什么都返回 0。
# manhattan 后没有括号，表示传递函数本身，供 search 内部按需调用。
results = []
for name, heuristic in (("Dijkstra", lambda node: 0),
                        ("A*", manhattan)):
    # 接收三个返回值：完整路径、总步数、展开次数。
    trace = []
    path, cost, expanded = search(heuristic, trace)
    results.append((name, heuristic, path, trace))
    # 先检查有路，再检查路径合法，最后检查是否达到本地图已知的最短代价 17。
    assert path, "Expected this fixed map to have a path"
    validate(path, cost)
    assert cost == 17
    # f 字符串把变量值填入文本。正常运行本例：
    # Dijkstra: cost=17, expanded=68
    # A*: cost=17, expanded=51
    # 这里 A* 展开更少；具体次数与地图、邻居顺序、平局处理方式有关。
    # expanded 是搜索工作量的一项指标，并不直接等于实际运行时间。
    print(f"{name}: cost={cost}, expanded={expanded}")
    # 转为集合，方便在打印每个格子时快速判断它是否属于最终路径。
    # 集合不保留路径顺序，但画地图只需要判断“在不在路径上”。
    path_nodes = set(path)
    # 外层循环逐行处理 y，内层循环逐列处理 x，最后拼成一行字符串打印。
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            node = (x, y)
            # 连续条件表达式按以下顺序挑选符号：
            # S=起点，G=终点，#=障碍物，*=路径上的其他格子，.=空闲格子。
            # 起终点也属于 path_nodes，但先判断它们，所以显示 S/G 而不是 *。
            symbol = ("S" if node == START else "G" if node == GOAL
                      else "#" if node in BLOCKED
                      else "*" if node in path_nodes else ".")
            row.append(symbol)
        # "".join(row) 将这一行的字符连接起来，中间不添加分隔符。
        print("".join(row))
    # 打印空行，分隔两种算法的结果。
    print()

# 两种算法的文字结果输出后，打开左右并排的动画窗口。
# plt.show() 在通常的桌面脚本运行方式下会等待窗口关闭，随后程序结束。
animation = visualize(results)

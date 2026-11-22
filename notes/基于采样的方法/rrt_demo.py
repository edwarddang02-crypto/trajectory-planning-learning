"""二维基础 RRT：碰撞检查、路径验证，以及 Matplotlib 树生长动画。

直接运行：python rrt_demo.py
只计算并保存结果图（不弹窗）：python rrt_demo.py --no-show
动画回放 seed=0 中通过碰撞检查的扩展边，失败的尝试不会画入树中。
"""

import argparse
import csv
import math
import random
from pathlib import Path
from statistics import median

# 机器人简化成半径为 0.2 的圆；规划节点表示这个圆的中心。
RADIUS = 0.2
# 原始地图为 [0, 10] × [0, 10]，中心与墙至少保持一个机器人半径。
LOW, HIGH = RADIUS, 10.0 - RADIUS
START, GOAL = (1.0, 5.0), (9.0, 5.0)
# 每个障碍物用 (圆心 x, 圆心 y, 半径) 表示。
OBSTACLES = [(5.0, 5.0, 1.6), (3.0, 7.5, 0.8), (7.0, 2.5, 0.8)]


def edge_free(a, b):
    """判断机器人中心从 a 沿直线运动到 b 的全过程是否安全。"""
    # 地图边界是凸矩形：两端都在范围内，整条线段也在范围内。
    if any(not (LOW <= x <= HIGH and LOW <= y <= HIGH)
           for x, y in (a, b)):
        return False
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    for cx, cy, radius in OBSTACLES:
        # 将圆心投影到直线，再把参数 t 限制在 [0, 1]，得到线段最近点。
        # 零长度线段退化成一个点，也可用来检查起终点是否合法。
        t = 0.0 if length_sq == 0 else (
            (cx - a[0]) * dx + (cy - a[1]) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        closest = (a[0] + t * dx, a[1] + t * dy)
        # 将障碍物半径增大 RADIUS，等价于把机器人中心当成质点检查。
        # 接触也算碰撞；微小容差用于避免浮点误差。
        if math.dist(closest, (cx, cy)) <= radius + RADIUS + 1e-10:
            return False
    return True


def plan(seed, step=0.5, goal_bias=0.1, budget=3000, trace=None):
    """返回 (路径, 已用采样次数, 树节点数)；预算耗尽时路径为空。

    seed 固定随机序列；step 为每次扩展的最大距离；goal_bias 为直接
    朝终点采样的概率；budget 为最大采样次数（包括扩展失败的尝试）。
    trace 可传入空列表，用于记录动画；省略它不影响原来的调用方式。
    """
    rng = random.Random(seed)
    # nodes[i] 是节点坐标，parents[i] 是父节点下标；根节点没有父节点。
    nodes, parents = [START], [-1]
    if not edge_free(START, START) or not edge_free(GOAL, GOAL):
        raise ValueError("Start or goal is invalid")
    for iteration in range(1, budget + 1):
        # 1. 采样：多数时候随机探索，少数时候直接用终点引导生长。
        target = GOAL if rng.random() < goal_bias else (
            rng.uniform(LOW, HIGH), rng.uniform(LOW, HIGH))
        # 2. 最近邻：在已有树节点中寻找距离采样目标最近的节点。
        nearest = min(range(len(nodes)),
                      key=lambda i: math.dist(nodes[i], target))
        a = nodes[nearest]
        distance = math.dist(a, target)
        if distance < 1e-12:
            continue
        # 3. 扩展：向目标最多走 step；目标更近时直接走到目标。
        fraction = min(1.0, step / distance)
        new = (a[0] + fraction * (target[0] - a[0]),
               a[1] + fraction * (target[1] - a[1]))
        # 4. 检查整条新边；两端安全仍可能在中途穿过障碍物。
        if not edge_free(a, new):
            continue
        nodes.append(new)
        parents.append(nearest)
        if trace is not None:
            trace.append((a, new, iteration))
        # 5. 新节点接近终点时，额外检查到终点的最后一段。
        if math.dist(new, GOAL) <= step and edge_free(new, GOAL):
            path = [GOAL]
            index = len(nodes) - 1
            # 沿父节点从末端回溯到根，再反转为“起点 → 终点”。
            while index != -1:
                if math.dist(path[-1], nodes[index]) > 1e-12:
                    path.append(nodes[index])
                index = parents[index]
            path.reverse()
            return path, iteration, len(nodes)
    return [], budget, len(nodes)


def verify(path):
    """检查路径端点与每条边，返回各段欧氏距离之和。"""
    assert path[0] == START and path[-1] == GOAL
    for a, b in zip(path, path[1:]):
        assert edge_free(a, b)
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    assert length >= math.dist(START, GOAL) - 1e-9
    return length


def visualize(path, trace, used, count, show=True):
    """保存 seed=0 的最终图；弹窗时逐边回放树，再显示最终路径。"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Rectangle

    # 图中文字用英文，避免本机没有中文字体时出现方框；代码注释为中文。
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.add_patch(Rectangle((LOW, LOW), HIGH - LOW, HIGH - LOW,
                           fill=False, edgecolor="gray", linestyle="--",
                           label="Robot-center boundary"))
    for i, (cx, cy, radius) in enumerate(OBSTACLES):
        ax.add_patch(Circle((cx, cy), radius, color="0.3",
                            label="Obstacle" if i == 0 else None))
        ax.add_patch(Circle((cx, cy), radius + RADIUS, fill=False,
                            edgecolor="darkorange", linestyle="--",
                            label="Obstacle + robot radius" if i == 0 else None))

    tree = LineCollection([], colors="steelblue", linewidths=0.8,
                          alpha=0.7, label="Accepted tree edges")
    ax.add_collection(tree)
    route, = ax.plot([], [], color="crimson", linewidth=2.5,
                     label="Final path", zorder=4)
    ax.scatter(*START, color="green", s=80, label="Start", zorder=5)
    ax.scatter(*GOAL, color="red", marker="*", s=140, label="Goal", zorder=5)
    tip, = ax.plot([], [], "o", color="dodgerblue", markersize=5)
    ax.legend(loc="upper right", fontsize=8)
    title = ax.set_title("")
    fig.tight_layout()

    def update(frame):
        # frame 表示已经显示多少条有效边，不等于采样次数。
        tree.set_segments([(a, b) for a, b, _ in trace[:frame]])
        if frame:
            new = trace[frame - 1][1]
            tip.set_data([new[0]], [new[1]])
        else:
            tip.set_data([], [])
        route.set_data([], [])
        iteration = trace[frame - 1][2] if frame else 0
        status = f"iteration={iteration}, nodes={frame + 1}"
        if frame == len(trace):
            if path:
                route.set_data(*zip(*path))
                length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
                status = f"FOUND | length={length:.3f}, iterations={used}, nodes={count}"
            else:
                status = f"NO PATH | iterations={used}, nodes={count}"
        title.set_text(f"Basic RRT | seed=0 | {status}")
        return tree, route, tip, title

    # 先保存完整结果；随后从空树开始回放，避免截图只保存到动画中间。
    update(len(trace))
    output = Path(__file__).resolve().with_name("rrt_seed0.png")
    fig.savefig(output, dpi=160)
    print(f"Saved figure: {output}")
    if show:
        animation = FuncAnimation(fig, update, frames=range(len(trace) + 1),
                                  init_func=lambda: update(0), interval=60,
                                  repeat=False, blit=False)
        # 局部变量保留动画对象，直到阻塞式窗口关闭，防止动画被提前回收。
        plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true",
                        help="Save the final plot without opening an animation window")
    args = parser.parse_args()
    # 保留原来的 20 次实验；只记录 seed=0 的动画，便于重复观察同一次规划。
    lengths, iterations = [], []
    demo_trace = []
    demo = None
    for seed in range(20):
        path, used, count = plan(seed, trace=demo_trace if seed == 0 else None)
        if seed == 0:
            demo = (path, used, count)
        if not path:
            print(f"seed={seed:02d}: no path within budget, nodes={count}")
            continue
        length = verify(path)
        lengths.append(length)
        iterations.append(used)
        print(f"seed={seed:02d}: length={length:.3f}, "
              f"iterations={used}, nodes={count}")
        if seed == 0:
            with Path("rrt_path_seed0.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows(path)
    print(f"success={len(lengths)}/20")
    if lengths:
        print(f"median length={median(lengths):.3f}")
        print(f"median iterations={median(iterations):.1f}")
    visualize(demo[0], demo_trace, demo[1], demo[2], show=not args.no_show)

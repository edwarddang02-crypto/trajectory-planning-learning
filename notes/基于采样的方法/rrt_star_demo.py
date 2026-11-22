"""二维 RRT*：中文注释、树的重新接线动画，以及路径长度收敛曲线。

运行：python rrt_star_demo.py
不弹窗：python rrt_star_demo.py --no-show
地图与 rrt_demo.py 相同；找到首条路径后继续优化到 3000 次采样。
"""

import argparse
import csv
import math
import random
from pathlib import Path
from statistics import median

# 所有坐标均为圆形机器人的中心；障碍物格式为 (圆心 x, 圆心 y, 半径)。
RADIUS = 0.2
LOW, HIGH = RADIUS, 10.0 - RADIUS
START, GOAL = (1.0, 5.0), (9.0, 5.0)
OBSTACLES = [(5.0, 5.0, 1.6), (3.0, 7.5, 0.8), (7.0, 2.5, 0.8)]


def edge_free(a, b):
    """检查整条线段；将障碍物膨胀一个机器人半径来考虑机器人尺寸。"""
    if any(not (LOW <= x <= HIGH and LOW <= y <= HIGH)
           for x, y in (a, b)):
        return False
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    for cx, cy, radius in OBSTACLES:
        # 圆心在线段上的最近点：先投影到直线，再将参数截断到 [0, 1]。
        t = 0.0 if length_sq == 0 else (
            (cx - a[0]) * dx + (cy - a[1]) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        closest = (a[0] + t * dx, a[1] + t * dy)
        if math.dist(closest, (cx, cy)) <= radius + RADIUS + 1e-10:
            return False
    return True


def get_path(nodes, parents, index):
    """从可连接终点的节点沿父节点回溯，返回起点到终点的路径。"""
    path = [GOAL]
    while index != -1:
        if math.dist(path[-1], nodes[index]) > 1e-12:
            path.append(nodes[index])
        index = parents[index]
    path.reverse()
    return path


def plan(seed, step=0.5, goal_bias=0.1, budget=3000, trace=None):
    """返回 (预算结束时的最佳路径, 已用采样次数, 树节点数)。

    step 限制向采样点扩展的距离；选父节点和重新接线可连接更远的邻居。
    邻域半径为 min(2, 12*sqrt(log(n)/n))，随二维树的节点数增加而收缩。
    本例优化欧氏路径长度，不涉及运动时间、速度或动力学约束。
    trace 可传空列表，记录动画快照；不记录时可减少内存使用。
    """
    if not edge_free(START, START) or not edge_free(GOAL, GOAL):
        raise ValueError("Start or goal is invalid")
    rng = random.Random(seed)
    # 三组下标一一对应：节点位置、父节点、从起点到节点的累计路程。
    nodes, parents, costs = [START], [-1], [0.0]
    # children 用于重新接线后向下更新整个子树的代价。
    children = [set()]
    goal_candidates = []
    path, best_cost = [], math.inf
    first_iteration = None
    rewires = 0
    changed = set()

    def snapshot(iteration):
        if trace is not None:
            # 必须复制当前树，后续 rewiring 会改变父节点与累计代价。
            trace.append(dict(iteration=iteration, nodes=tuple(nodes),
                              parents=tuple(parents), costs=tuple(costs),
                              path=tuple(path), cost=best_cost,
                              first_iteration=first_iteration, rewires=rewires,
                              changed=tuple(changed)))
        changed.clear()

    snapshot(0)
    for iteration in range(1, budget + 1):
        # 1. 采样、寻找最近邻、向目标走至多 step，与基础 RRT 相同。
        target = GOAL if rng.random() < goal_bias else (
            rng.uniform(LOW, HIGH), rng.uniform(LOW, HIGH))
        nearest = min(range(len(nodes)),
                      key=lambda i: math.dist(nodes[i], target))
        a = nodes[nearest]
        distance = math.dist(a, target)
        if distance < 1e-12:
            continue
        fraction = min(1.0, step / distance)
        new = (a[0] + fraction * (target[0] - a[0]),
               a[1] + fraction * (target[1] - a[1]))
        if not edge_free(a, new):
            continue

        # 2. 找邻居；最近节点作为可行的默认父节点，即使它不在邻域中。
        n = len(nodes) + 1
        radius = min(2.0, 12.0 * math.sqrt(math.log(n) / n))
        distances = [math.dist(p, new) for p in nodes]
        near = [i for i, d in enumerate(distances) if d <= radius]
        parent = nearest
        cost = costs[nearest] + distances[nearest]
        # RRT* 的第一处区别：选累计代价最低的可连接父节点。
        for i in near:
            candidate = costs[i] + distances[i]
            if candidate < cost and edge_free(nodes[i], new):
                parent, cost = i, candidate
        new_index = len(nodes)
        nodes.append(new)
        parents.append(parent)
        costs.append(cost)
        children.append(set())
        children[parent].add(new_index)

        # 3. RRT* 的第二处区别：邻居改走新节点更便宜时，重新接线。
        for i in near:
            candidate = costs[new_index] + distances[i]
            if candidate + 1e-12 >= costs[i] or not edge_free(new, nodes[i]):
                continue
            children[parents[i]].remove(i)
            parents[i] = new_index
            children[new_index].add(i)
            rewires += 1
            changed.add(i)
            # 不能只改 i：它下面所有后代从起点出发的路程也一起变短。
            pending = [i]
            while pending:
                child = pending.pop()
                costs[child] = costs[parents[child]] + math.dist(
                    nodes[parents[child]], nodes[child])
                pending.extend(children[child])

        # 4. 保留所有可连接终点的节点；旧候选的代价也可能因重接线下降。
        if math.dist(new, GOAL) <= step and edge_free(new, GOAL):
            goal_candidates.append(new_index)
        first_found = False
        if goal_candidates:
            best = min(goal_candidates,
                       key=lambda i: costs[i] + math.dist(nodes[i], GOAL))
            best_cost = costs[best] + math.dist(nodes[best], GOAL)
            path = get_path(nodes, parents, best)
            if first_iteration is None:
                first_iteration = iteration
                first_found = True

        # 每约 20 次采样记录一帧，并额外记录首解；无需逐次保存整棵树。
        if iteration % 20 == 0 or first_found:
            snapshot(iteration)

    snapshot(budget)
    return path, budget, len(nodes)


def verify(path):
    """验证起终点和每条路径边，返回路径长度。"""
    assert path[0] == START and path[-1] == GOAL
    assert all(edge_free(a, b) for a, b in zip(path, path[1:]))
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    assert length >= math.dist(START, GOAL) - 1e-9
    return length


def visualize(trace, show=True):
    """回放当前树的真实连接关系，并同步显示首解之后的长度变化。"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Rectangle

    fig, (ax, curve) = plt.subplots(1, 2, figsize=(13, 6),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.add_patch(Rectangle((LOW, LOW), HIGH - LOW, HIGH - LOW,
                           fill=False, edgecolor="gray", linestyle="--"))
    for i, (cx, cy, radius) in enumerate(OBSTACLES):
        ax.add_patch(Circle((cx, cy), radius, color="0.3",
                            label="Obstacle" if i == 0 else None))
        ax.add_patch(Circle((cx, cy), radius + RADIUS, fill=False,
                            edgecolor="darkorange", linestyle="--",
                            label="Obstacle + robot radius" if i == 0 else None))
    tree = LineCollection([], colors="steelblue", linewidths=0.45, alpha=0.3)
    rewired = LineCollection([], colors="darkorange", linewidths=1.5,
                             label="Rewired since last frame")
    ax.add_collection(tree)
    ax.add_collection(rewired)
    route, = ax.plot([], [], color="crimson", linewidth=2.5,
                     label="Current best path", zorder=4)
    ax.scatter(*START, color="green", s=60, label="Start", zorder=5)
    ax.scatter(*GOAL, color="red", marker="*", s=100, label="Goal", zorder=5)
    ax.legend(loc="upper right", fontsize=7)
    curve.set(xlabel="Sampling iterations", ylabel="Best path length",
              title="Best-so-far cost (after first solution)",
              xlim=(0, max(1, trace[-1]["iteration"])))
    curve.grid(alpha=0.2)
    finite_costs = [s["cost"] for s in trace if math.isfinite(s["cost"])]
    if finite_costs:
        curve.set_ylim(min(finite_costs) - 0.2, max(finite_costs) + 0.2)
    convergence, = curve.plot([], [], color="crimson", drawstyle="steps-post")
    title = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    def update(frame):
        state = trace[frame]
        nodes, parents = state["nodes"], state["parents"]
        # 每帧重新读取父节点，旧边被删除、新边被画出，才能真实展示 rewiring。
        tree.set_segments([(nodes[p], nodes[i]) for i, p in enumerate(parents)
                           if p != -1])
        rewired.set_segments([(nodes[parents[i]], nodes[i])
                              for i in state["changed"]])
        route.set_data(*zip(*state["path"]) if state["path"] else ([], []))
        history = [s for s in trace[:frame + 1] if math.isfinite(s["cost"])]
        convergence.set_data([s["iteration"] for s in history],
                             [s["cost"] for s in history])
        status = f"best length={state['cost']:.3f}" if history else "searching"
        title.set_text(f"RRT* | seed=0 | iteration={state['iteration']} | "
                       f"nodes={len(nodes)} | {status}\n"
                       f"first solution iteration={state['first_iteration']} | "
                       f"total rewires={state['rewires']}")
        return tree, rewired, route, convergence, title

    update(len(trace) - 1)
    output = Path(__file__).resolve().with_name("rrt_star_seed0.png")
    fig.savefig(output, dpi=160)
    print(f"Saved figure: {output}")
    if show:
        animation = FuncAnimation(fig, update, frames=len(trace),
                                  init_func=lambda: update(0), interval=80,
                                  repeat=False, blit=False)
        # 保留动画引用，直到弹窗关闭。
        plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true",
                        help="Save the final plot without opening an animation window")
    args = parser.parse_args()
    lengths, demo_trace = [], []
    for seed in range(20):
        path, used, count = plan(seed, trace=demo_trace if seed == 0 else None)
        if not path:
            print(f"seed={seed:02d}: no path within budget, nodes={count}")
            continue
        length = verify(path)
        lengths.append(length)
        print(f"seed={seed:02d}: length={length:.3f}, "
              f"iterations={used}, nodes={count}")
        if seed == 0:
            output = Path(__file__).resolve().with_name("rrt_star_path_seed0.csv")
            with output.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows(path)
    print(f"success={len(lengths)}/20")
    if lengths:
        print(f"median length={median(lengths):.3f}")
    visualize(demo_trace, show=not args.no_show)

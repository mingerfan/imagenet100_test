"""
Boot插入优化器

类比加油站问题：
- 汽车最多跑level格需要加油（密文深度到level需要boot）
- 每一格的加油站费用不同（不同位置的boot成本 = boot_cost * ct_num）
- 目标：跑完整个计算图的最小boot总费用

使用动态规划求解全局最优解。
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class NodeInfo:
    """节点信息"""
    index: int  # 拓扑序
    name: str
    depth_delta: int  # 该节点增加的深度
    ct_num: int  # 密文数量
    op_type: str


class BootOptimizer:
    """使用动态规划的Boot优化器

    dp[i][d] = 处理到第i个节点，当前深度为d时的最小boot成本

    状态转移：
    1. 不在节点i后插入boot: dp[i+1][d + depth_delta] = dp[i][d]
    2. 在节点i后插入boot: dp[i+1][d + depth_delta - level] = dp[i][d] + ct_num * boot_cost
    """

    def __init__(self, level: int, boot_cost: float):
        """
        Args:
            level: 密文深度容量（通常为10）
            boot_cost: 单次boot的基础成本
        """
        self.level = level
        self.boot_cost = boot_cost

    def optimize(self, nodes: List[NodeInfo]) -> Tuple[Dict[int, int], float]:
        """使用动态规划优化boot插入

        Args:
            nodes: 按拓扑序排列的节点列表（不包含fused节点）

        Returns:
            (boot_plan, total_cost)
            - boot_plan: {node_index: boot_count} 表示在该节点之后插入的boot次数
            - total_cost: 最小总boot成本
        """
        if not nodes:
            return {}, 0.0

        n = len(nodes)

        # dp[i] 是一个字典: {depth: (cost, parent_info)}
        # parent_info = (prev_node_idx, prev_depth, boot_count)
        dp = [{} for _ in range(n + 1)]

        # 初始状态：深度0，成本0
        dp[0][0] = (0.0, None)

        # 动态规划转移
        for i in range(n):
            node = nodes[i]

            if not dp[i]:  # 当前状态为空，跳过
                continue

            for depth, (cost, _) in dp[i].items():
                next_depth = depth + node.depth_delta

                # 选项1：不在当前节点后插入boot（如果深度允许）
                if next_depth <= self.level:
                    if next_depth not in dp[i + 1] or dp[i + 1][next_depth][0] > cost:
                        dp[i + 1][next_depth] = (cost, (i, depth, 0))

                # 选项2：在当前节点后插入boot
                # 计算需要插入的boot次数（通常为1，但如果depth_delta很大可能需要多次）
                boots_needed = (next_depth - 1) // self.level + 1 if next_depth > self.level else 1

                for boot_count in range(1, boots_needed + 1):
                    depth_after_boot = next_depth - boot_count * self.level
                    if depth_after_boot < 0:
                        continue

                    boot_cost_here = node.ct_num * self.boot_cost * boot_count
                    new_cost = cost + boot_cost_here

                    if depth_after_boot not in dp[i + 1] or dp[i + 1][depth_after_boot][0] > new_cost:
                        dp[i + 1][depth_after_boot] = (new_cost, (i, depth, boot_count))

        # 找到最终状态的最小成本
        if not dp[n]:
            return {}, float('inf')

        min_cost = float('inf')
        final_depth = None
        for depth, (cost, _) in dp[n].items():
            if cost < min_cost:
                min_cost = cost
                final_depth = depth

        # 回溯重建boot方案
        boot_plan = {}
        curr_idx = n
        curr_depth = final_depth

        while curr_idx > 0 and curr_depth is not None:
            if curr_depth not in dp[curr_idx]:
                break

            _, parent_info = dp[curr_idx][curr_depth]
            if parent_info is None:
                break

            prev_idx, prev_depth, boot_count = parent_info

            if boot_count > 0:
                # 记录在prev_idx节点后插入boot
                boot_plan[prev_idx] = boot_plan.get(prev_idx, 0) + boot_count

            curr_idx = prev_idx
            curr_depth = prev_depth

        return boot_plan, min_cost


def test_boot_optimizer():
    """测试boot优化器"""
    # 模拟一个计算图
    nodes = [
        NodeInfo(0, "conv1", 2, 10, "conv"),
        NodeInfo(1, "relu1", 1, 10, "relu"),
        NodeInfo(2, "conv2", 2, 20, "conv"),  # ct=20，成本高
        NodeInfo(3, "relu2", 1, 20, "relu"),  # ct=20，成本高
        NodeInfo(4, "conv3", 2, 15, "conv"),
        NodeInfo(5, "relu3", 1, 15, "relu"),
        NodeInfo(6, "conv4", 2, 5, "conv"),   # ct=5，成本低
        NodeInfo(7, "relu4", 1, 5, "relu"),
    ]

    level = 10
    boot_cost = 98136

    # 动态规划算法（最优）
    optimizer = BootOptimizer(level, boot_cost)
    boot_plan, total_cost = optimizer.optimize(nodes)

    print("=== 动态规划最优解 ===")
    print(f"Boot插入方案:")
    for node_idx, boot_count in sorted(boot_plan.items()):
        node = nodes[node_idx]
        print(f"  在节点 {node.name} (index={node_idx}, ct={node.ct_num}) 后插入 {boot_count} 次boot")
    print(f"最小总成本: {total_cost:.2f}")

    # 对比：简单策略（每次深度超过level就在当前位置boot）
    print("\n=== 简单策略对比（每次深度>level就boot）===")
    simple_cost = 0
    depth = 0
    simple_boots = []
    for i, node in enumerate(nodes):
        depth += node.depth_delta
        if depth > level:
            simple_boots.append((i, node.name, node.ct_num))
            simple_cost += node.ct_num * boot_cost
            depth = depth - level

    for node_idx, name, ct in simple_boots:
        print(f"  在节点 {name} (index={node_idx}, ct={ct}) 后boot")
    print(f"简单策略总成本: {simple_cost:.2f}")

    print(f"\n=== 优化效果 ===")
    if simple_cost > 0:
        saving = (1 - total_cost / simple_cost) * 100
        print(f"成本节省: {saving:.1f}%")
        print(f"绝对节省: {simple_cost - total_cost:.2f}")
    else:
        print("简单策略无boot开销")

    # 验证深度约束
    print("\n=== 验证深度约束 ===")
    depth = 0
    for i, node in enumerate(nodes):
        depth += node.depth_delta
        if i in boot_plan:
            print(f"节点 {node.name}: 深度 {depth} -> boot -> 深度 {depth - boot_plan[i] * level}")
            depth -= boot_plan[i] * level
        else:
            print(f"节点 {node.name}: 深度 {depth}")

        if depth > level:
            print(f"  警告：深度超过level！")


if __name__ == "__main__":
    test_boot_optimizer()

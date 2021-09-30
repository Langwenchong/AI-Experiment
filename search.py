"""Implementation of the A* algorithm.

This file contains a skeleton implementation of the A* algorithm. It is a single
method that accepts the root node and runs the A* algorithm
using that node's methods to generate children, evaluate heuristics, etc.
This way, plugging in root nodes of different types, we can run this A* to
solve different problems.

"""


def Astar(root):
    """Runs the A* algorithm given the root node. The class of the root node
    defines the problem that's being solved. The algorithm either returns the solution
    as a path from the start node to the goal node or returns None if there's no solution.

    Parameters
    ----------
    root: Node
        The start node of the problem to be solved.

    Returns
    -------
        path: list of Nodes or None
            The solution, a path from the initial node to the goal node.
            If there is no solution it should return None
    """

    # TODO: add your code here
    # Some helper pseudo-code:
    # 1. Create an empty fringe and add your root node (you can use lists, sets, heaps, ... )
    # 2. While the container is not empty:
    # 3.      Pop the best? node (Use the attribute `node.f` in comparison)
    # 4.      If that's a goal node, return node.get_path()
    # 5.      Otherwise, add the children of the node to the fringe
    # 6. Return None
    #
    # Some notes:
    # You can access the state of a node by `node.state`. (You may also want to store evaluated states)
    # You should consider the states evaluated and the ones in the fringe to avoid repeated calculation in 5. above.
    # You can compare two node states by node1.state == node2.state

    # 可以展开的节点状态
    openList = [root]
    # 不可以再展开的节点状态
    closeList = []
    # 遍历可展开节点
    while openList:
        # 首先是寻找最小f的节点进行展开
        minIdx = 0
        for i in range(len(openList)):
            if openList[i].f < openList[minIdx].f:
                minIdx = i
        # 检测这个f最小的节点
        exploreNode = openList.pop(minIdx)
        # 如果还未展开过
        if exploreNode._get_state() not in closeList:
            # 将其加入到closeList中
            closeList.append(exploreNode._get_state())
            # 如果是目的节点状态，结束，调用get_path()返还路径信息
            if exploreNode.is_goal():
                return exploreNode.get_path()
            else:
                # 否则不是目的节点，将其展开的子节点加入到openList中
                openList.extend(exploreNode.generate_children())
    # openList的所有节点均已展开仍未到达目的节点，无解
    return None

    pass

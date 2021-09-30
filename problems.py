from node import Node
import copy


class FifteensNode(Node):
    """Extends the Node class to solve the 15 puzzle.

    Parameters
    ----------
    parent : Node, optional
        The parent node. It is optional only if the input_str is provided. Default is None.

    g : int or float, optional
        The cost to reach this node from the start node : g(n).
        In this puzzle it is the number of moves to reach this node from the initial configuration.
        It is optional only if the input_str is provided. Default is 0.

    board : list of lists
        The two-dimensional list that describes the state. It is a 4x4 array of values 0, ..., 15.
        It is optional only if the input_str is provided. Default is None.

    input_str : str
        The input string to be parsed to create the board.
        The argument 'board' will be ignored, if input_str is provided.
        Example: input_str = '1 2 3 4\n5 6 7 8\n9 10 0 11\n13 14 15 12' # 0 represents the empty cell

    Examples
    ----------
    Initialization with an input string (Only the first/root construction call should be formatted like this):
    >>> n = FifteensNode(input_str=initial_state_str)
    >>> print(n)
      5  1  4  8
      7     2 11
      9  3 14 10
      6 13 15 12

    Generating a child node (All the child construction calls should be formatted like this) ::
    >>> n = FifteensNode(parent=p, g=p.g+c, board=updated_board)
    >>> print(n)
      5  1  4  8
      7  2    11
      9  3 14 10
      6 13 15 12

    """

    def __init__(self, parent=None, g=0, board=None, input_str=None):
        # NOTE: You shouldn't modify the constructor
        if input_str:
            self.board = []
            for i, line in enumerate(filter(None, input_str.splitlines())):
                self.board.append([int(n) for n in line.split()])
        else:
            self.board = board

        super(FifteensNode, self).__init__(parent, g)

    def generate_children(self):
        """Generates children by trying all 4 possible moves of the empty cell.

        Returns
        -------
            children : list of Nodes
                The list of child nodes.
        """

        # TODO: add your code here
        # You should use self.board to produce children. Don't forget to create a new board for each child
        # e.g you can use copy.deepcopy function from the standard library.

        # 主要思路就是找到那个空节点(根据提示这里就是数值为0的节点)
        # 然后将其上下左右都枚举一次生成下一代可能出现的子情况棋盘
        # 要注意边界条件问题，不能让0节点出棋盘
        # 因此需要考虑0节点的索引值是否处于棋盘的边界

        # 首先我们要先寻找到0节点的位置
        # 初始化0节点的标记索引
        zero_i = 0
        zero_j = 0

        # 遍历棋盘，之所以不写成4*4，是因为棋盘大小不固定
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 0:
                    # 找到了0节点
                    # 更新索引值
                    zero_i = i
                    zero_j = j

        # 接下来我们需要上下左右移动0节点生成子状态棋盘
        # 为了存储所有的子节点，我们创建一个列表
        children = []

        # 下面逐一上下左右移动0节点

        # 向左移动
        # 首先考虑边界值问题，必须保证0节点向左移动不会出棋盘
        if zero_j != 0:
            # 首先子棋盘的值分布先初始化Wie父节点状态
            childBoard = copy.deepcopy(self.board)
            # 0节点向左和左边的棋子交换位置
            childBoard[zero_i][zero_j] = childBoard[zero_i][zero_j-1]
            childBoard[zero_i][zero_j-1] = 0
            # 将生成的子节点加入到child列表中
            # 要注意要标记父节点，还有g实际上就是在父节点状态基础上—+1,因为每次只移动一次0节点
            childNode1 = FifteensNode(
                parent=self, g=self.g+1, board=childBoard)
            children.append(childNode1)

        # 向右移动
        if zero_j != 3:
            childBoard = copy.deepcopy(self.board)
            childBoard[zero_i][zero_j] = childBoard[zero_i][zero_j+1]
            childBoard[zero_i][zero_j+1] = 0
            childNode2 = FifteensNode(
                parent=self, g=self.g+1, board=childBoard)
            children.append(childNode2)

        # 向上移动
        if zero_i != 0:
            childBoard = copy.deepcopy(self.board)
            childBoard[zero_i][zero_j] = childBoard[zero_i-1][zero_j]
            childBoard[zero_i-1][zero_j] = 0
            childNode3 = FifteensNode(
                parent=self, g=self.g+1, board=childBoard)
            children.append(childNode3)

        # 向下移动
        if zero_i != 3:
            childBoard = copy.deepcopy(self.board)
            childBoard[zero_i][zero_j] = childBoard[zero_i+1][zero_j]
            childBoard[zero_i+1][zero_j] = 0
            childNode4 = FifteensNode(
                parent=self, g=self.g+1, board=childBoard)
            children.append(childNode4)

        return children
        pass

    def is_goal(self):
        """Decides whether this search state is the final state of the puzzle.

        Returns
        -------
            is_goal : bool
                True if this search state is the goal state, False otherwise.
        """

        # TODO: add your code here
        # You should use self.board to decide.

        if (self.board[0][0] == 1) and (self.board[0][1] == 2) and (self.board[0][2] == 3) and (self.board[0][3] == 4) and (self.board[1][0] == 5) and (self.board[1][1] == 6) and (self.board[1][2] == 7) and (self.board[1][3] == 8) and (self.board[2][0] == 9) and (self.board[2][1] == 10) and (self.board[2][2] == 11) and (self.board[2][3] == 12) and (self.board[3][0] == 13) and (self.board[3][1] == 14) and (self.board[3][2] == 15) and (self.board[3][3] == 0):
            return True
        else:
            return False
        pass

    def evaluate_heuristic(self):
        """Heuristic function h(n) that estimates the minimum number of moves
        required to reach the goal state from this node.

        Returns
        -------
            h : int or float
                The heuristic value for this state.
        """

        # TODO: add your code here
        # You may want to use self.board here.

        # 用曼哈顿距离估计h值
        h = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                # 求解每一个非0节点的到目的节点状态的曼哈顿距离
                # 因此我们要使用self棋盘的每一个棋子的索引值和目的棋盘每一个棋子的索引值来求解曼哈顿距离
                # 我们可以使用(value-1)/4个value%4-1来获取目的棋盘的每一个棋子的索引值
                if self.board[i][j] != 0:
                    # 行间距
                    distance_i = abs(i - int((self.board[i][j]-1)/4))
                    # 列间距
                    if(self.board[i][j] % 4 == 0):
                        distance_j = abs(j-3)
                    else:
                        distance_j = abs(j-(self.board[i][j] % 4-1))
                    h = h+distance_i+distance_j
        return h
        pass

    def _get_state(self):
        """Returns an hashable representation of this search state.

        Returns
        -------
            state: tuple
                The hashable representation of the search state
        """
        # NOTE: You shouldn't modify this method.
        return tuple([n for row in self.board for n in row])

    def __str__(self):
        """Returns the string representation of this node.

        Returns
        -------
            state_str : str
                The string representation of the node.
        """
        # NOTE: You shouldn't modify this method.
        sb = []  # String builder
        for row in self.board:
            for i in row:
                sb.append(' ')
                if i == 0:
                    sb.append('  ')
                else:
                    if i < 10:
                        sb.append(' ')
                    sb.append(str(i))
            sb.append('\n')
        return ''.join(sb)


class SuperqueensNode(Node):
    """Extends the Node class to solve the Superqueens problem.

    Parameters
    ----------
    parent : Node, optional
        The parent node. Default is None.

    g : int or float, optional
        The cost to reach this node from the start node : g(n).
        In this problem it is the number of pairs of superqueens that can attack each other in this state configuration.
        Default is 1.

    queen_positions : list of pairs
        The list that stores the x and y positions of the queens in this state configuration.
        Example: [(q1_y,q1_x),(q2_y,q2_x)]. Note that the upper left corner is the origin and y increases downward
        Default is the empty list [].
        ------> x
        |
        |
        v
        y

    n : int
        The size of the board (n x n)

    Examples
    ----------
    Initialization with a board size (Only the first/root construction call should be formatted like this):
    >>> n = SuperqueensNode(n=4)
    >>> print(n)
         .  .  .  .
         .  .  .  .
         .  .  .  .
         .  .  .  .

    Generating a child node (All the child construction calls should be formatted like this):
    >>> n = SuperqueensNode(parent=p, g=p.g+c, queen_positions=updated_queen_positions, n=p.n)
    >>> print(n)
         Q  .  .  .
         .  .  .  .
         .  .  .  .
         .  .  .  .

    """

    def __init__(self, parent=None, g=0, queen_positions=[], n=1):
        # NOTE: You shouldn't modify the constructor
        self.queen_positions = queen_positions
        self.n = n
        super(SuperqueensNode, self).__init__(parent, g)

    def generate_children(self):
        """Generates children by adding a new queen.

        Returns
        -------
            children : list of Nodes
                The list of child nodes.
        """
        # TODO: add your code here
        # You should use self.queen_positions and self.n to produce children.
        # Don't forget to create a new queen_positions list for each child.
        # You can use copy.deepcopy function from the standard library.

        # 首先我们要知道根据题干已经保证了Q一定是在不同的行和列之间
        # 因此我们只需要考虑两个对角线上的冲突以及其实'日'字形的冲突
        # 首先我们要一次放置Q，只需要根据题干保证每次在一个新的列上放置Q
        # 同时要保证新放置的Q和之前的所有Q不在统一行上面
        children = []

        # 首先我们要得知下一个要放置Q的列数
        # queen_positions是棋盘中放置的Q的坐标列表
        col = len(self.queen_positions)

        # 然后我们寻找一个不同行的位置放置这个Q
        for row in range(self.n):
            ok = True
            # 遍历Q的坐标列表
            for Q in self.queen_positions:
                # 如果这个row和某一个已经放置的Q的行坐标相同，那么这里就不能放置
                if Q[0] == row:
                    ok = False
                    break

            # 如果这个row是合法可以放置新的Q
            if ok:
                # 首先我们将这个新的Q放置到棋盘上
                new_queen_positions = copy.deepcopy(self.queen_positions)
                # 将新放置的Q的坐标加入到列表中
                new_queen_positions.append((row, col))

                # 此时我们已经生成了一个新棋盘，我们需要统计此时这个子棋盘的冲突个数
                # 首先子棋盘的冲突个数一定大于等于父棋盘的冲突个数
                new_g = self.g

                i = row
                j = col
                while i > 0 and j > 0:
                    # 首先我们考虑自左上到右下方向的对角线上的冲突
                    i = i-1
                    j = j-1
                    if (i, j) in self.queen_positions:
                        # 找到一个新的冲突，因此冲突个数加一
                        new_g = new_g+1
                    # 我们要注意之所以只考虑i-1,j-1往左上蔓延的方向，而不考虑往右下蔓延的方向
                    # 是因为这个新放置的Q的右方还没有Q呢，因此其右方一定是没有冲突的

                i = row
                j = col
                while i < self.n and j > 0:
                    # 再检测自右上到左下的对角线上的冲突
                    i = i+1
                    j = j-1
                    if (i, j) in self.queen_positions:
                        new_g = new_g+1
                    # 同理我们只检测向左下蔓延的方向的新冲突，因为右侧一定是没有冲突的

                # 我们再检测'日'字型的冲突
                # 我们要注意j一定是减的，因为我们还是只需要检测向左侧蔓延的'日'字形
                # 因为新放置的Q的右侧是没有其他Q的，因此一定是没有冲突的
                i = row-2
                j = col-1
                if(i, j) in self.queen_positions:
                    new_g = new_g+1
                i = row-1
                j = col-2
                if(i, j) in self.queen_positions:
                    new_g = new_g+1
                i = row+2
                j = col-1
                if(i, j) in self.queen_positions:
                    new_g = new_g+1
                i = row+1
                j = col-2
                if(i, j) in self.queen_positions:
                    new_g = new_g+1

                # 将这个新的子棋盘的状态加到child列表中
                childNode = SuperqueensNode(
                    parent=self, g=new_g, queen_positions=new_queen_positions, n=self.n)
                children.append(childNode)
        return children
        pass

    def is_goal(self):
        """Decides whether all the queens are placed on the board.

        Returns
        -------
            is_goal : bool
                True if all the queens are placed on the board, False otherwise.
        """
        # You should use self.queen_positions and self.n to decide.
        # TODO: add your code here

        return len(self.queen_positions) == self.n
        pass

    def evaluate_heuristic(self):
        """Heuristic function h(n) that estimates the minimum number of conflicts required to reach the final state.

        Returns
        -------
            h : int or float
                The heuristic value for this state.
        """
        # If you want to design a heuristic for this problem, you should use self.queen_positions and self.n.
        # TODO: add your code here (optional)
        # 不需要增加新的h来估计了，已经在生成子状态的函数中统计了
        return 0

    def _get_state(self):
        """Returns an hashable representation of this search state.

        Returns
        -------
            state: tuple
                The hashable representation of the search state
        """
        # NOTE: You shouldn't modify this method.
        return tuple(self.queen_positions)

    def __str__(self):
        """Returns the string representation of this node.

        Returns
        -------
            state_str : str
                The string representation of the node.
        """
        # NOTE: You shouldn't modify this method.
        sb = [[' . '] * self.n for i in range(self.n)]  # String builder
        for i, j in self.queen_positions:
            sb[i][j] = ' Q '
        return '\n'.join([''.join(row) for row in sb])

## 介绍文档

> 这是人工智能A*算法的一个应用例题，他通过两个案例：①八数码问题和②最少冲突n皇后问题来帮助我们熟悉A\*算法的应用。以下是操作手册。

**1 Introduction**

In this assignment, you will implement A* algorithm in Python, and apply it to the problems below. The helper code you need is provided on the course web page, you just need to fill in the missing parts.

The layout of the helper code:

• node.py - The implementation of the Node class. (Do not modify this file!)

• problems.py - The partial implementation of the FifteensNode and SuperqueensNode classes.

• search.py - The skeleton of the A* algorithm.

• test.py - A helper script that runs some tests.

You need to code up the function Astar in search.py, and the following methods of the FifteensNode and SuperqueensNode classes:

• is_goal

• generate_children

• evaluate_heuristic

You can use built-in functions & data structures, and the standard library, but you **cannot** use/import anyone else’s code or any package that is not in the Python standard library. You can use test.py to test your code:

```python
python -m unittest test.py
```

However, your code will be tested on some secret instances of the problems; therefore, you should be careful about the boundary cases.

Note that you should not necessarily expect your algorithms to solve every instance (difficult instances may require too much time or memory; that does not mean that you did not solve the experiment problem correctly). Of course, your code is expected to output the right answer when it outputs something.

**2 Fifteens Puzzle**

The first problem that you will solve using A* is the classic fifteens puzzle (the four-by-four version of the eights puzzle studied in class) where you can move any horizontally or vertically adjacent tile into the empty slot.

For example, here is the goal state:

![](https://gitee.com/Langwenchong/figure-bed/raw/master/20210930100140.png)

The following states are one move away from the goal state:

![](https://gitee.com/Langwenchong/figure-bed/raw/master/20210930100159.png)

Every move has a cost of 1, and of course, since you are using A*, your program should find the optimal (lowest-cost) solution. In principle, there may be more than one optimal solution; your program is just expected to give one of these optimal solutions—it does not matter which one. You can use the heuristics discussed in the lectures, or, if you want, you can try to design something even better (the book has some more detail).

**3 Superqueens Puzzle**

Consider a modified chess piece that can move like a queen, but also like a knight. We will call such a piece a “superqueen” (it is also known as an “amazon”). This leads to a new “superqueens” puzzle. We formulate the puzzle as a constraint optimization problem: each row and each column can have at most one superqueen (that’s a hard constraint), and we try to minimize the number of pairs of superqueens that attack each other (either diagonally or with a knight’s move).

For example, the following is an optimal solution for a 7 by 7 board: there are 3 total attacks (two diagonal, one a knight’s move).

![](https://gitee.com/Langwenchong/figure-bed/raw/master/20210930100351.png)

The cost of a node should be the number of pairs of superqueens that attack each other so far; you can set the heuristic to 0. You are not required to implement sophisticated variable ordering (that is, you can always consider the variables in the same order), constraint propagation, etc. (although you are welcome to do so)
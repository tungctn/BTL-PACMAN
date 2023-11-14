# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

# question 1
def depthFirstSearch(problem: SearchProblem):
    stack = util.Stack()
    visited = set()

    # Đưa nút gốc vào Stack cùng với đường đi trống
    stack.push((problem.getStartState(), []))

    while not stack.isEmpty():
        # Lấy ra nút và đường đi từ Stack
        current_state, path = stack.pop()
        # Kiểm tra nếu đã đến đích
        if problem.isGoalState(current_state):
            return path

        # Nếu chưa thăm nút này, mở rộng và thêm vào Stack
        if current_state not in visited:
            visited.add(current_state)

            # Lấy các nút con và thêm vào Stack
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:
                    new_path = path + [action]
                    stack.push((successor, new_path))

    # Trường hợp không tìm thấy đường đi
    # return []
    util.raiseNotDefined()

# question 2
def breadthFirstSearch(problem: SearchProblem):
    queue = util.Queue()
    visited = set()
    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        state, path = queue.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)
            for successor, action, _ in problem.getSuccessors(state):
                new_path = path + [action]
                queue.push((successor, new_path))

    util.raiseNotDefined()

# question 3
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priorityQueue = util.PriorityQueue()
    priorityQueue.push(problem.getStartState(), 0)

    # Set chứa các nút đã thăm
    visited = set()

    # Lưu đường đi và chi phí tương ứng
    paths_and_costs = {}
    paths_and_costs[problem.getStartState()] = ([], 0)

    while not priorityQueue.isEmpty():
        # Lấy ra nút có chi phí nhỏ nhất
        current_state = priorityQueue.pop()

        # Kiểm tra nếu đã đến đích thì trả về đường đi
        if problem.isGoalState(current_state):
            return paths_and_costs[current_state][0]

        # Nếu chưa thăm nút này, mở rộng và thêm vào hàng đợi ưu tiên
        if current_state not in visited:
            visited.add(current_state)
            
            # Lấy các nút con và thêm vào hàng đợi ưu tiên
            for successor, action, cost in problem.getSuccessors(current_state):
                new_cost = paths_and_costs[current_state][1] + cost
                new_path = paths_and_costs[current_state][0] + [action]
                
                # Nếu nút con chưa được thăm hoặc có chi phí nhỏ hơn thì thêm vào hàng đợi ưu tiên
                if (
                    successor not in paths_and_costs
                    or new_cost < paths_and_costs[successor][1]
                ):
                    paths_and_costs[successor] = (new_path, new_cost)
                    priorityQueue.push(successor, new_cost)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# question 4
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    queue = util.PriorityQueue()
    visited = set()
    startState = problem.getStartState()
    queue.push((problem.getStartState(), [], 0), 0)

    print("Start State:", problem.getStartState())
    print("Adding to Queue:", (startState, [], 0), "with priority", 0)
    print("Visiting State:", visited)

    while not queue.isEmpty():
        state, path, cost = queue.pop()
        # print("\nCurrent State:", state)
        # print("Current Path:", path)
        # print("Current Cost:", cost)

        if problem.isGoalState(state):
            print("Goal State Reached!")
            return path
        
        # Nếu chưa thăm nút này, mở rộng và thêm vào hàng đợi ưu tiên
        if state not in visited:
            visited.add(state)
            print("Visiting State:", state)
            
            # Lấy các nút con và thêm vào hàng đợi ưu tiên
            for successor, action, stepCost in problem.getSuccessors(state):
                new_path = path + [action]
                new_cost = cost + stepCost
                heuristic_cost = heuristic(successor, problem)
                total_cost = new_cost + heuristic_cost

                # print("  Successor State:", successor)
                # print("  Action:", action)
                # print("  Step Cost:", stepCost)
                # print("  New Path:", new_path)
                # print("  New Cost:", new_cost)
                # print("  Heuristic Cost:", heuristic_cost)
                print("  Total Cost:", total_cost)

                queue.push((successor, new_path, new_cost), total_cost)

    print("No path found to the goal.")

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

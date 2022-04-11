# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

# from calendar import c
from cmath import inf
# from sys import maxsize
# from turtle import pos
# from numpy import argmax, inner, zeros
from numpy import argmax, zeros
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        self.gameState = startingGameState

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        return (self.startingPosition, [])

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        isGoal = True
        for idx, pos in enumerate(self.corners):
            if pos not in state[1]:
                isGoal = False
                break
        return isGoal
    
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            
            "*** YOUR CODE HERE ***"
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            corner_state = state[1]

            if not hitsWall:
                for idx, pos in enumerate(self.corners):
                    if (nextx, nexty) == pos and (nextx, nexty) not in corner_state:
                        nextState = ((nextx, nexty), corner_state+[(nextx, nexty)])
                        break
                    else:
                        nextState = ((nextx, nexty), corner_state)
                successors.append((nextState, action, 1))

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    height, width = walls.height-2, walls.width-2
    corner_state = []       
    cost = 1e9
    for corner in corners:
        if corner not in state[1]:
            corner_state.append(corner)
            cost = min(cost, util.manhattanDistance(state[0], corner))

    if len(corner_state) == 4:
        return cost + 2*(min(height, width)-1) + (max(height, width)-1)

    if len(corner_state) == 3:
        return cost + (min(height, width)-1) + (max(height, width)-1)

    if len(corner_state) == 2:
        return cost + util.manhattanDistance(corner_state[0], corner_state[1])

    if len(corner_state) == 1:
        return cost
    
    return 0


    # cost = 0
    # maxCost = 0

    # corner_state = state[1]
    # for pos in corners:
    #     if pos not in corner_state: 
    #         cost = mazeDistance(pos, state[0], problem.gameState)
    #         # cost = util.manhattanDistance(pos, state[0])
    #         maxCost = max(maxCost, cost)
    
    # return maxCost


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

        """我的额外参数"""
        top, right = self.walls.height, self.walls.width
        ifFood = startingGameState.getFood()
        # print('height', top)
        # print('width', right)
        # 获取初始食物的位置，建立初始食物对编号的映射，位置对编号的映射
        siteMapping = {}
        foodMapping = {}
        
        for x in range(1, right-1):
            for y in range(1, top-1):
                if ifFood[x][y] == True:
                    foodMapping[(x, y)] = len(foodMapping)  # 食物对编号的映射
                if not self.walls[x][y]:
                    siteMapping[(x, y)] = len(siteMapping)  # 位置对编号的映射
        foods = list(foodMapping.keys())
        sites = list(siteMapping.keys())
        self.heuristicInfo['foodMapping'] = foodMapping
        self.heuristicInfo['siteMapping'] = siteMapping
        # print(foodMapping)
        # print(foods)

        # print(ifFood, '\n')
        # print(self.walls, '\n')
        # for i in range(top-1, -1, -1):
        #     s = ''
        #     for j in range(right):
        #         s += 'T' if self.walls[j][i] else 'F'
        #     print(s)
        # 可行位置到糖的曼哈顿距离
        dist2 = zeros((len(sites), len(foods)))
        # for outer in range(len(sites)):
        #     for inner in range(len(foods)):
        #         dist2[outer][inner] = util.manhattanDistance((sites[outer][0], sites[outer][1]), (foods[inner][0], foods[inner][1]))
        # print('Dist2', dist2)
        for site in sites:
            siteIdx = siteMapping[site]
            tmpDist = self.breadFirst(site)
            # print(site, tmpDist)
            for food in foods:
                foodIdx = foodMapping[food]
                dist2[siteIdx][foodIdx] = tmpDist[food]
        # print('Dist2', dist2)
        self.heuristicInfo['dist2'] = dist2
        

        # 算出各个糖之间的曼哈顿距离
        dist1 = zeros((len(foods), len(foods)))
        # for outer in range(len(foods)):
        #     for inner in range(len(foods)):
        #         dist1[outer][inner] = util.manhattanDistance((foods[outer][0], foods[outer][1]), (foods[inner][0], foods[inner][1]))
        #     dist1[outer][outer] = inf
        # print('Dist1', dist1)
        for foodOuter in foods:
            outerIdx = foodMapping[foodOuter]
            outerSiteIdx = siteMapping[foodOuter]
            for foodInner in foods:
                innerIdx = foodMapping[foodInner]
                dist1[outerIdx][innerIdx] = dist2[outerSiteIdx][innerIdx]
            dist1[outerIdx][outerIdx] = inf
        self.heuristicInfo['dist1'] = dist1
        # print('Dist1\n', dist1)
        self.heuristicInfo['count'] = {'minSum': 0, 'maxSingle': 0, 'maxSinglePlus': 0}
    
    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost
    
    def ownGetSuccessors(self, state):
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def breadFirst(self, startPosition):
        self.foods = self.startingGameState.getFood()
        self.walls = self.startingGameState.getWalls()
        top, right = self.walls.height, self.walls.width
        
        positions = util.Queue()
        route = []
        distMapping = {(x, y): inf for x in range(1, right-1) for y in range(1, top-1)}
        
        positions.push(startPosition)
        distMapping[startPosition] = 0
        
        while not positions.isEmpty():
            current_position = positions.pop()
            # print('current_position: ', current_position)
            if current_position not in route:
                route.append(current_position)
                # print('next')
                for position, direction, value in self.ownGetSuccessors([current_position, self.foods]):
                    positions.push(position[0])
                    distMapping[position[0]] = min(distMapping[position[0]], distMapping[current_position]+1)
                    # print(position[0], distMapping[position[0]])
                
        # exit()
        return distMapping

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    top, right = problem.walls.height, problem.walls.width
    positionIdx = problem.heuristicInfo['siteMapping'][position]
    foodMapping = problem.heuristicInfo['foodMapping']
    dist1 = problem.heuristicInfo['dist1']
    dist2 = problem.heuristicInfo['dist2']

    foods = []
    cost1 = inf
    cost2 = 0

    nearest = (-1, -1)      # “距离”糖豆人最近的糖豆的“距离”
    nearestDist = inf
    for x in range(1, right): 
        for y in range(1, top):
            if foodGrid[x][y] == True:      # 用TRUE表示有糖，False表示无糖
                foodIdx = foodMapping[(x, y)]
                if dist2[positionIdx][foodIdx] < nearestDist:
                    nearest = (x, y)
                    nearestDist = dist2[positionIdx][foodIdx]
                foods.append((x, y))
    cost1 = nearestDist     # 没有豆子时，cost2此时为inf
    cost2 = nearestDist     # 没有豆子时，cost2此时为inf
  
    # 记录到从糖i(i=1,...,n)到指定糖j的“距离”最小值
    # min Sum
    foodsIdx = [foodMapping[food] for food in foods]
    tmp = [0]
    for food in foods:
        idx = foodMapping[food]
        if food != nearest:
            tmp.append(min(dist1[idx][foodsIdx]))
        else:
            cost1 += min(min(dist1[idx][foodsIdx]), 0)
    cost1 += sum(tmp) - max(tmp)
    cost1 = cost1 if cost1 != inf else 0

    # 记录豆子之间的单个最大“距离”，并记录最大距离对应的豆子编号
    maxSingle = 0
    records = []
    for i in foodsIdx:
        for j in foodsIdx:
            if j == i:
                continue
            # maxSingle = max(maxSingle, dist1[i][j])
            if dist1[i][j] >= maxSingle:
                maxSingle = dist1[i][j]
                records.append((i, j))
    
    # 如果存在两个及以上豆子，则 cost2 = maxSingle + 糖豆人现有豆子的最小距离
    # 如果只有一个豆子，则 coct2 = 糖豆人到豆子的最小距离
    # 如果没有豆子，则 cost2 = 0
    cost2 += maxSingle
    cost2 = cost2 if cost2 != inf else 0


    # 如果存在两个及以上豆子，则 cost3 = maxSingle + 糖豆人到maxSingle对应豆子的最小距离
        # (如果有多对豆子之间的距离同为最大值，则这些豆子都是maxSingle对应豆子)
    # 如果只有一个豆子，则 coct3 = 糖豆人到豆子的最小距离
    # 如果没有豆子，则 cost3 = 0
    cost3 = 0
    cost3 += maxSingle
    t_min = inf
    for record in records:
        t_min = min(t_min, min(dist2[positionIdx][record[0]], dist2[positionIdx][record[1]]))
    cost3 += t_min      
    if cost3 == inf:        # 此时如果cost3为inf，说明地图上只存在1个豆子或0个豆子
        if nearestDist != inf:  # 对应还有1个豆子
            cost3 = nearestDist 
        else:                   # 对应没有豆子
            cost3 = 0           

    cost = [cost1, cost2, cost3]
    problem.heuristicInfo['count']['minSum']        += argmax(cost) == 0
    problem.heuristicInfo['count']['maxSingle']     += argmax(cost) == 1
    problem.heuristicInfo['count']['maxSinglePlus'] += argmax(cost) == 2

    # print(problem.heuristicInfo['count'])
    return max(cost)

    
    # 以下部分是俊鸿的一些思路
    # top, right = problem.walls.height-2, problem.walls.width-2
    # cost = 0
    # mincost = inf
    # minx = right
    # maxx = 0
    # miny = top
    # maxy = 0
    # for foodx in range(1, right+1):
    #     for foody in range(1, top+1):
    #         if foodGrid[foodx][foody] == 1:
    #             cost = max(cost, util.manhattanDistance((foodx,foody),position))
    #             minx = min(foodx,minx)
    #             maxx = max(foodx,maxx)
    #             miny = min(foody,miny)
    #             maxy = max(foody,maxy)
    #             if util.manhattanDistance((foodx,foody),position) <= mincost:
    #                 mincost = util.manhattanDistance((foodx,foody),position)
    
    # cost2 = max(cost,state[1].count()-1+mincost,maxx - minx + maxy - miny + mincost)
    # if mincost == inf:
    #     cost2 = max(cost,state[1].count(),maxx - minx + maxy - miny +1)

    # return max(cost1, cost2)

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        moves_quene = util.Queue()
        moves = []
        positions = util.Queue()
        expend_nodes = []
        
        positions.push(startPosition)
        moves_quene.push([])
        
        while not positions.isEmpty():
            current_position = positions.pop()
            moves = moves_quene.pop()
            
            if current_position not in expend_nodes:
                expend_nodes.append(current_position)

                if problem.isGoalState(current_position):
                    return moves
                
                for position, direction, cost in problem.getSuccessors(current_position):
                    positions.push(position)
                    moves_quene.push(moves + [direction])
        # util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y] == True
        # util.raiseNotDefined()

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))

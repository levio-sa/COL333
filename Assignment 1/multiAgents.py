#Chirag Jain 2019CS10342
#Sarthak Singla 2019CS10397

# multiAgents.py
# --------------
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


from pacman import SCARED_TIME
from util import manhattanDistance,Queue
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minus_inf=-99999999
        if action=='Stop':# less prefferred
            return successorGameState.getScore()
        
        Ghost_score=0.0
        for ghoststates in newGhostStates:
            d=manhattanDistance(newPos,ghoststates.getPosition())
            if ghoststates.scaredTimer==0 and d<=1:
                return minus_inf# highly negative case 
            elif d<=ghoststates.scaredTimer:# eating ghost possible for points
                Ghost_score+=1*ghoststates.scaredTimer/float(d)
            else:# ghost cannot be eaten or the ghost is unscared
                Ghost_score+=-1/float(d)
        
        Food_score=0.0
        Food_count=(500-len(newFood.asList()))/500.0
        for food in newFood.asList():
            d=manhattanDistance(food,newPos)
            if d<=5:#if food in range near us go towards it
                Food_score+=2*Food_count/float(d)
            else:#if food away from us less preffered
                Food_score+=Food_count/float(d)
        
        return successorGameState.getScore()+Food_score+Ghost_score
        
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        bestval=-float('inf')
        bestaction=None
        agentIndex=0
        cdepth=0
        
        for action in gameState.getLegalActions(agentIndex):
            newgamestate=gameState.generateSuccessor(agentIndex, action)
            val=self.getval(newgamestate,agentIndex+1,cdepth) 
            if bestval < val:
                bestaction=action
                bestval=val

        return bestaction
        util.raiseNotDefined()

    def getval(self,gamestate,agentIndex,cdepth):
        if agentIndex==gamestate.getNumAgents():
            agentIndex=0#Pacman Index
            cdepth+=1#depth increase by 1
        
        #         depth reached  Terminal state win      Terminal state lose
        if cdepth==self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate)
        
        bestval=None
        if agentIndex==0:#Move Pacman
            bestval=-float('inf')# max hence take least value
            for action in gamestate.getLegalActions(agentIndex):
                newgamestate=gamestate.generateSuccessor(agentIndex, action)
                val=self.getval(newgamestate,agentIndex+1,cdepth)
                bestval=max( bestval,val )
        else:#Move Ghost
            bestval=float('inf')#min hence take greatest value
            for action in gamestate.getLegalActions(agentIndex):
                newgamestate=gamestate.generateSuccessor(agentIndex, action)
                val=self.getval(newgamestate,agentIndex+1,cdepth)
                bestval=min( bestval,val )

        return bestval

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestval=-float('inf')
        bestaction=None
        agentIndex=0
        cdepth=0
        alpha,beta=-float('inf'),float('inf')
        
        for action in gameState.getLegalActions(agentIndex):
            newgamestate=gameState.generateSuccessor(agentIndex, action)
            val=self.getval(newgamestate,agentIndex+1,cdepth,alpha,beta) 
            if bestval < val:
                bestaction=action
                bestval=val
            alpha=max(alpha,bestval)# no pruning possible as this is root node
        
        return bestaction
        util.raiseNotDefined()

    def getval(self,gamestate,agentIndex,cdepth,alpha,beta):
        if agentIndex==gamestate.getNumAgents():
            agentIndex=0
            cdepth+=1
        #         depth reached  Terminal state win      Terminal state lose
        if cdepth==self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate)
        
        bestval=None
        if agentIndex==0:#Move Pacman
            bestval=-float('inf')
            for action in gamestate.getLegalActions(agentIndex):
                newgamestate=gamestate.generateSuccessor(agentIndex, action)
                val=self.getval(newgamestate,agentIndex+1,cdepth,alpha,beta)
                bestval=max( bestval,val )
                alpha=max(alpha,bestval)
                if bestval>beta:#prune the tree
                    break
        else:#Move Ghost
            bestval=float('inf')
            for action in gamestate.getLegalActions(agentIndex):
                newgamestate=gamestate.generateSuccessor(agentIndex, action)
                val=self.getval(newgamestate,agentIndex+1,cdepth,alpha,beta)
                bestval=min( bestval,val )
                beta=min(beta,bestval)
                if alpha>bestval:#prune the tree
                    break

        return bestval

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestval=-float('inf')
        bestaction=None
        agentIndex=0
        cdepth=0
        for action in gameState.getLegalActions(agentIndex):
            newgamestate=gameState.generateSuccessor(agentIndex, action)
            val=self.getval(newgamestate,agentIndex+1,cdepth) 
            if bestval < val:
                bestaction=action
                bestval=val

        return bestaction
        util.raiseNotDefined()
    
    def getval(self,gamestate,agentIndex,cdepth):
        if agentIndex==gamestate.getNumAgents():
            agentIndex=0
            cdepth+=1
        if cdepth==self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate)
        
        bestval=None
        if agentIndex==0:#Move Pacman
            bestval=-float('inf')
            for action in gamestate.getLegalActions(agentIndex):
                newgamestate=gamestate.generateSuccessor(agentIndex, action)
                val=self.getval(newgamestate,agentIndex+1,cdepth)
                bestval=max( bestval,val )
        else:#Move Ghost
            bestval=0.0
            actions=gamestate.getLegalActions(agentIndex)
            if len(actions)==0:#if ghost is trapped in walls
                return self.getval(gamestate,agentIndex+1,cdepth)
            probability=1.0/len(actions)# probability of a random legal action
            for action in actions:
                newgamestate=gamestate.generateSuccessor(agentIndex, action)
                val=self.getval(newgamestate,agentIndex+1,cdepth)
                bestval += probability*val

        return bestval

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    If          there is a scared ghost run towards closest scared ghost
    else if     there is a capsule run towards closest capsule
    else        run towards closest food

    high negative award for moving in 2 squares radius near the unscared ghost
    high negative award for losing the game
    """
    M_INF=-999999999
    P_INF= 999999999
    # Food 10pts
    # Ghost Eaten 200pts
    # WIN 500 pts
    # LOSE -500pts
    # Capsule 0pts only change ghost to scared
    # Each Move -1pt (living cost)
    Terminal_Score=0.0
    if currentGameState.isWin():
        Terminal_Score=0
    if currentGameState.isLose():
        Terminal_Score=M_INF

    closestfood, closestcapsule, ghostdist=BFS(currentGameState)
   
    Ghost_Score=0.0
    Scared_Ghost_Score=0.0
    newGhostStates = currentGameState.getGhostStates()
    for i in range(len(newGhostStates)):
        ghost = newGhostStates[i]
        d=ghostdist[i]
        if d==None:
            continue
        elif ghost.scaredTimer==0:
            if d<=2:
                Ghost_Score+= -M_INF/float(d+1)
        elif ghost.scaredTimer!=0:
            Scared_Ghost_Score=max(Scared_Ghost_Score,100 + (10/d))

    if closestcapsule==None or Scared_Ghost_Score!=0:
        Capsule_Score=0.0
    else:
        Capsule_Score=1.0/closestcapsule

    if closestfood==None or Capsule_Score!=0 or Scared_Ghost_Score!=0:
        Food_Score=0.0
    else:
        Food_Score=2.0/closestfood

    return (
            100*
            (currentGameState.getScore() 
            + Food_Score
            + 2*Capsule_Score
            + Scared_Ghost_Score
            + 10*Ghost_Score
            + 1000*Terminal_Score )

            + 0.001*random.random()/2
            )

#user defined function
def BFS(currentGameState):
    "returns closest food, capsule and distace of all ghosts"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()    
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    widthm,heightm=newFood.width,newFood.height

    visited = [[0 for i in range(heightm)] for j in range(widthm)]
    
    closestfood = None
    ghostdist = [None for i in range(len(newGhostStates))]
    closestcapsule = None

    q = Queue()
    q.push((newPos,0))
    #print(newFood)
    #print(newPos)
    #print(widthm,heightm)
    visited[newPos[0]][newPos[1]]=1

    itemsfound = 0

    itemstofind = 1 # as food is always there
    if(len(newCapsules)!=0):#find closest capsule
        itemstofind+=1
    itemstofind+=len(newGhostStates)#find all ghosts dist

    while(not q.isEmpty()):
        curpos, distance = q.pop()
        allowedposlist = [(curpos[0]-1,curpos[1]), (curpos[0]+1,curpos[1]), (curpos[0],curpos[1]-1), (curpos[0],curpos[1]+1)]
        for pos in allowedposlist:
            if(pos[0]>=0 and pos[0]<widthm and pos[1]>=0 and pos[1]<heightm and not currentGameState.hasWall(pos[0],pos[1]) and not visited[pos[0]][pos[1]]):
                if(closestfood==None and currentGameState.hasFood(pos[0],pos[1])):
                    closestfood = distance+1
                    itemsfound+=1
                if(closestcapsule==None and pos in newCapsules):
                    closestcapsule = distance+1
                    itemsfound+=1
                for i in range(len(newGhostStates)):
                    ghost=newGhostStates[i]
                    if(ghostdist[i]==None and ghost.getPosition()==pos):
                        ghostdist[i] = distance+1
                        itemsfound+=1
                if(itemsfound==itemstofind):
                    return (closestfood, closestcapsule, ghostdist)
                
                q.push((pos,distance+1))
                visited[pos[0]][pos[1]]=1

    return (closestfood,closestcapsule,ghostdist)


# Abbreviation
better = betterEvaluationFunction

#Chirag Jain 2019CS10342
#Sarthak Singla 2019CS10397
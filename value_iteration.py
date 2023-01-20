import numpy as np

ACTIONS = ('U', 'R', 'D', 'L')

GAMMA = 0.8

DELTA_THRESHOLD = 1e-3

class Grid():
    def __init__(self, xSize, ySize, start_point):
        self.xSize = xSize
        self.ySize = ySize
        self.currX = start_point[0]
        self.currY = start_point[1]

    def set(self, actions, rewards):
        self.actions = actions
        self.rewards = rewards

    def move(self, action):
        if action in self.actions[(self.currX, self.currY)]:
            if action == 'U':
                self.currY+=1
            elif action == 'D':
                self.currY-=1
            elif action == 'R':
                self.currX+=1
            elif action == 'L':
                self.currX-=1
        return self.rewards.get((self.currX, self.currY),0)

    def set_state(self, state):
        self.currX = state[0]
        self.currY = state[1]

    def current_state(self):
        return (self.currX,self.currY)

    def all_state(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

def init_grid():
    grid = Grid(4,3, (0,0))

    rewards = {(3,1):-1, (3,2):1}
    actions = {
        (0,0):("U", "R"),
        (1,0):("L", "R"),
        (2,0):("U", "R", "L"),
        (3,0):("U", "L"),
        (0,1):("U", "D"),
        (2,1):("U", "R", "D"),
        (0,2):("D", "R"),
        (1,2):("L", "R"),
        (2,2):("D", "R", "L")
    }
    grid.set(actions, rewards)
    return grid

def print_values(V, grid):

    for y in reversed(range(grid.ySize)):
        print("----------------------------")
        for x in range(grid.xSize):
            value = V.get((x,y),0)
            print(" %.2f |" %value, end = "")
        print("\n")


def print_policy(policy, grid):
    for y in reversed(range(grid.ySize)):
        print("----------------")
        for x in range(grid.xSize):
            action = policy.get((x,y), ' ')
            print(" %s |" %action, end = "")
        print("\n")

if __name__ == '__main__':
    
    grid = init_grid()

    policy = {}
    
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTIONS)

    print("\n 초기정책 : ")
    print_policy(policy, grid)    

    V = {}
    states = grid.all_state()
    for s in states:
        if s in grid.rewards.keys():
            V[s] = 0
        else:
            V[s] = np.random.random()

    print("\n 초기 value function : ")
    print_values(V, grid)

    i=0
    while True:
        maxChange = 0
        for s in grid.actions.keys():
            oldValue = V[s]
            newValue = float('-inf')
            for a in ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                value = r + GAMMA * V[(grid.currX, grid.currY)]
                if newValue<value:
                    newValue = value
            V[s] = newValue
            maxChange = max(maxChange, abs(oldValue-V[s]))
        
        print("%i번째" %i)
        print_values(V,grid)
        i+=1
        if maxChange<DELTA_THRESHOLD:
            break

    for s in grid.actions.keys():
        actions = grid.actions[s]
        action = policy[s]
        value = float('-inf')
        for a in actions:
            grid.set_state(s)
            grid.move(a)
            if V[grid.current_state()] > value:
                value =  V[grid.current_state()]
                action = a
        policy[s] = action

    print_policy(policy, grid)
import numpy as np 
import random 
import itertools 
import matplotlib.pyplot as plt 

class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name, direction):
        
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size 
        self.intensity = intensity 
        self.channel = channel 
        self.reward = reward 
        self.name = name
        self.direction = direction 

class gameEnv():
    def __init__(self, partial, size, object_size):
        self.sizeX = size 
        self.sizeY = size 
        self.object_size = object_size 
        self.actions = 5 ## up, down, left, right, wait
        self.objects = []
        self.partial = partial 
        
    def reset(self):
        self.objects = []
        self.holeDir = []
        hero = gameOb((0, self.sizeY-1), self.object_size, 1, 2, None,'hero', None)
        self.objects.append(hero)
        bug = gameOb((self.sizeX-1, 0), self.object_size, 1, 1, 1, 'goal', None)
        self.objects.append(bug)
        
        ##### Add 5 obstacles #####
        for i in range(0, 5):
            direction = np.random.randint(0,4)
            hole = gameOb(self.newPosition(), self.object_size, 1, 0, -1, 'fire', direction)
            self.objects.append(hole)
            
        state = self.renderEnv()
        self.state = state 
        return state 
    
    def moveChar(self, direction):
        hero = self.objects[0]
        heroX = hero.x 
        heroY = hero.y 
        penalize = 0.0
        if direction==0 and hero.y>=1:  ## up
            hero.y -= 1
        if direction==1 and hero.y<=self.sizeY-2: ## down 
            hero.y += 1 
        if direction==2 and hero.x >= 1: ## left 
            hero.x -= 1 
        if direction==3 and hero.x <= self.sizeX - 2:  ## right
            hero.x += 1
        if direction == 4: ## wait
            pass 
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0 ## doesn't move 
        self.objects[0] = hero 
        return penalize 
    
    def moveHole(self):
        for obj in self.objects:
            if obj.name=='fire':
                direction = obj.direction 
                if direction == 0:
                    if (obj.y >= 1):
                        obj.y -= 1 
                if direction == 1:
                    if (obj.y <= self.sizeY-2):
                        obj.y += 1 
                if direction == 2:
                    if (obj.x >= 1):
                        obj.x -= 1 
                if direction == 3:
                    if (obj.x <= self.sizeX-2):
                        obj.x += 1
    
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]
    
    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj 
            else:
                others.append(obj)
            for other in others:
                if hero.x == other.x and hero.y == other.y:
                    return other.reward, False  ## done is always false
        return 0.0, False        
    def renderEnv(self):
        a = np.zeros([self.sizeY, self.sizeX, 3])
        hero = None 
        for item in self.objects:
            a[item.y:item.y+item.size, item.x:item.x+item.size, item.channel] = item.intensity 
        ## partial would always be false 
        b = a[:,:,0]
        c = a[:,:,1]
        d = a[:,:,2]
        a = np.stack([b, c, d], axis=2)
        return a 
    
    def step(self, action):
        penalty = self.moveChar(action)
        self.moveHole()
        reward, done = self.checkGoal()
        state = self.renderEnv()
        return state, (reward+penalty), done 

# env = gameEnv(partial=False, size=16, object_size=1)
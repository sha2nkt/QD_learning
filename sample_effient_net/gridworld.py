import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt


class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name,direction):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        self.direction = direction
        
class gameEnv():
    def __init__(self,partial,size, object_size):
        self.sizeX = size
        self.sizeY = size
        self.object_size = object_size
        self.actions = 5
        self.objects = []
        self.partial = partial
        a = self.reset()
        # plt.imshow(a,interpolation="nearest")
        
        
    def reset(self):
        self.objects = []
        self.holeDir = []
        hero = gameOb((0, self.sizeY-1),self.object_size,1,2,None,'hero',None)
        self.objects.append(hero)
        bug = gameOb((self.sizeX-1, 0),self.object_size,1,1,1,'goal',None)
        self.objects.append(bug)

        # 0 - up, 1 - down, 2 - left, 3 - right
        dir1 = np.random.randint(0,4)
        hole1 = gameOb(self.newPosition(),self.object_size,1,0,-1,'fire',dir1)
        self.objects.append(hole1)
        dir2 = np.random.randint(0,4)
        hole2 = gameOb(self.newPosition(),self.object_size,1,0,-1,'fire',dir2)
        self.objects.append(hole2)
        dir3 = np.random.randint(0,4)
        hole3 = gameOb(self.newPosition(),self.object_size,1,0,-1,'fire',dir3)
        self.objects.append(hole3)
        dir4 = np.random.randint(0,4)
        hole4 = gameOb(self.newPosition(),self.object_size,1,0,-1,'fire',dir4)
        self.objects.append(hole4)
        state = self.renderEnv()
        self.state = state
        return state

    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right, 4 - wait
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1   
        if direction == 4:
            # do nothing 
            pass
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize

    def moveHole(self):
        for obj in self.objects:
            if obj.name == 'fire':
                direction = obj.direction
                if direction == 0:
                    if(obj.y >= 1):
                        obj.y -= 1
                    else:
                        self.objects.remove(obj)
                        dir1 = np.random.randint(0,4)                        
                        self.objects.append(gameOb(self.sidePosition(),self.object_size,1,0,-1,'fire',dir1))
                if direction == 1:
                    if(obj.y <= self.sizeY-2):
                        obj.y += 1
                    else:
                        self.objects.remove(obj)
                        dir1 = np.random.randint(0,4)                        
                        self.objects.append(gameOb(self.sidePosition(),self.object_size,1,0,-1,'fire',dir1))
                if direction == 2:
                    if(obj.x >= 1):
                        obj.x -= 1
                    else:
                        self.objects.remove(obj)
                        dir1 = np.random.randint(0,4)                        
                        self.objects.append(gameOb(self.sidePosition(),self.object_size,1,0,-1,'fire',dir1))                    
                if direction == 3:
                    if(obj.x <= self.sizeX-2):
                        obj.x += 1
                    else:
                        self.objects.remove(obj)
                        dir1 = np.random.randint(0,4)                        
                        self.objects.append(gameOb(self.sidePosition(),self.object_size,1,0,-1,'fire',dir1))
                       

    def sidePosition(self):
        # when one object disappears, another object appears on the side
        iterables = [[0,self.sizeX], [0,self.sizeY]]
        points = []
        x_idxs = self.sizeX
        y_idxs = self.sizeY
        for y_idx in range(y_idxs):
            points.append([0,y_idx])
        for y_idx in range(y_idxs):
            points.append([x_idxs, y_idx])
        for x_idx in range(x_idxs):
            points.append([x_idx, 0])
        for x_idx in range(x_idxs):
            points.append([x_idx, y_idxs])

        # for t in itertools.product(*iterables):
        #     points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            try:
                points.remove(pos)
            except ValueError:
                pass
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]



    
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]


    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                #self.objects.remove(other)
                if other.reward == 1:
                    #self.objects.append(gameOb(self.newPosition(),1,1,1,1,'goal'))
                    #do nothing for now
                    pass
                else:
                    # dir1 = np.random.randint(0,4)
                    # print(dir1)
                    # self.objects.append(gameOb(self.newPosition(),1,1,0,-1,'fire',dir1))
                    pass
                return other.reward,False
        if ended == False:
            return 0.0,False

    def renderEnv(self):
        a = np.zeros([self.sizeY,self.sizeX,3])
        # a = np.ones([self.sizeY,self.sizeX,3])
        # a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y:item.y+item.size,item.x:item.x+item.size,item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]
        b = a[:,:,0]
        c = a[:,:,1]
        d = a[:,:,2]
        a = np.stack([b,c,d],axis=2)
        return a

    def step(self,action):
        penalty = self.moveChar(action)
        self.moveHole()
        reward,done = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
        #     print(done)
        #     print(reward)
        #     print(penalty)
            return state,(reward+penalty),done
        else:
            return state,(reward+penalty),done
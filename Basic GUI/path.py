from vector2d import Vector2D
from scat import Scat
from copy import copy
from line import Line
from math import sqrt



class Path(object):

    def __init__(self, path = [], scat = 0, arrive = False, time = 0):
        self.path = path
        path.append(scat)
        self.arrive = arrive
        self.distance = time
        self.end = False

    def makePaths(self, extraPaths = 0, destination = 0, scats=[],xM=0):
        speed = 1/60
        temp = []
        j = 0
        for node in self.path:
            i = 0
            for scat in scats:
                if (node == scat.SCAT):
                    break
                i += 1
            if (node != destination):
                for neighbour in scats[i].neighbours:
                    if (extraPaths == 0):
                        break
                    l = 0
                    for scat in scats:
                        if (neighbour == scat.SCAT):
                            break
                        l += 1
                    if (j==0):
                        if (neighbour != destination and neighbour != self.path[0] and neighbour != self.path[j+1]):
                            tempAr = [self.path[0]]
                            m = 0
                            for scat in scats:
                                if (self.path[0] == scat.SCAT):
                                    break
                                m += 1
                            temp.append(Path(copy(tempAr), copy(int(neighbour)), False, copy(sqrt((scats[l].pos.x - scats[m].pos.x)*(scats[l].pos.x - scats[m].pos.x)+(scats[l].pos.y - scats[m].pos.y)*(scats[l].pos.y - scats[m].pos.y))*xM/speed)))
                            extraPaths -= 1
                    elif (neighbour != destination and neighbour != self.path[0] and neighbour != self.path[j-1] and neighbour != self.path[j+1]):
                        k = 0
                        tempAr = []
                        tempDis = 0
                        while (k < j+1):
                            tempAr.append(self.path[k])
                            if (k < j):
                                m = 0
                                for scat in scats:
                                    if (self.path[k] == scat.SCAT):
                                        break
                                    m += 1
                                n = 0
                                for scat in scats:
                                    if (self.path[k+1] == scat.SCAT):
                                        break
                                    n += 1
                                tempDis += sqrt((scats[m].pos.x - scats[n].pos.x)*(scats[m].pos.x - scats[n].pos.x)+(scats[m].pos.y - scats[n].pos.y)*(scats[m].pos.y - scats[n].pos.y))
                            k += 1
                        m = 0
                        for scat in scats:
                            if (self.path[k] == scat.SCAT):
                                break
                            m += 1
                        temp.append(Path(copy(tempAr), copy(int(neighbour)), False, copy(tempDis + 30 + sqrt((scats[l].pos.x - scats[m].pos.x)*(scats[l].pos.x - scats[m].pos.x)+(scats[l].pos.y - scats[m].pos.y)*(scats[l].pos.y - scats[m].pos.y)))))
                        extraPaths -= 1
            j += 1
        return temp
    
    def search(self, destination = 0, scats=[],xM=0):
        speed = 1/60
        for scat in scats:
            scat.fSCAT = float(0000)
            scat.distance = 0
            if (float(self.path[len(self.path)-1]) == scat.SCAT):
                current = scat
        searchList = []
        searching = True
        while(searching):
            for neighbour in current.neighbours:
                visited = False
                for scat in self.path:
                    if (scat == neighbour):
                        visited = True
                        break
                nbr = current
                for scat in scats:
                    if (neighbour == scat.SCAT):
                        nbr = scat
                        break
                if (visited == False):
                    tempDist = current.distance + 30 + copy(sqrt((current.pos.x - nbr.pos.x)*(current.pos.x - nbr.pos.x)+(current.pos.y - nbr.pos.y)*(current.pos.y - nbr.pos.y))*xM/speed)
                    if (nbr.distance == 0):
                        nbr.fSCAT = copy(current.SCAT)
                        nbr.distance = copy(tempDist)
                        searchList.append(nbr)
                    elif (nbr.distance > tempDist):
                        nbr.fSCAT = copy(current.SCAT)
                        nbr.distance = copy(tempDist)
            if (len(searchList) == 0):
                break
            searchList.sort(key=lambda x: x.distance, reverse=False)
            current = searchList.pop(0)
            if (current.SCAT == destination):
                searching = False
                self.arrive = True
                self.distance += current.distance
                tempPath = []
                tempSearch = True
                while (tempSearch):
                    tempPath.insert(0,current.SCAT)
                    for scat in scats:
                        if (current.fSCAT == scat.SCAT):
                            current = scat
                            break
                    if (float(current.SCAT) == float(self.path[len(self.path)-1])):
                        tempSearch = False
                        self.path = self.path + tempPath




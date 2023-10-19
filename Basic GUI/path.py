from vector2d import Vector2D
from scat import Scat
from copy import copy
from line import Line



class Path(object):

    def __init__(self, path = [], scat = 0, arrive = False, time = 0):
        self.path = path
        path.append(float(scat))
        self.arrive = arrive
        self.distance = time
        self.end = False


    def step(self, destination = 0, scats=[], lines=[]):
        current = self.path[len(self.path)-1]
        i = 0
        temp = -1
        temp2 = []
        for scat in scats:
            if (current == scat.SCAT):
                break
            i += 1
        for neighbour in scats[i].neighbours:
            
            visited = False
            for scat in self.path:
                if (scat == neighbour):
                    visited = True
                    break
            if (visited == False):
                if (temp == -1):
                    temp = neighbour
                else:
                    for line in lines:
                        if (line.SCAT1 == self.path[len(self.path)-1] or line.SCAT2 == self.path[len(self.path)-1]):
                            if (line.SCAT1 == neighbour or line.SCAT2 == neighbour):
                                if (neighbour == destination):
                                    temp2.append(Path(copy(self.path),copy(neighbour),True,copy(self.distance) + copy(line.distance)))
                                else:
                                    temp2.append(Path(copy(self.path),copy(neighbour),False,copy(self.distance) + copy(line.distance)))
                                break
        if (temp != -1):
            for line in lines:
                if (line.SCAT1 == self.path[len(self.path)-1] or line.SCAT2 == self.path[len(self.path)-1]):
                    if (line.SCAT1 == temp or line.SCAT2 == temp):
                        if (temp == destination):
                            self.arrive = True
                            self.path.append(temp)
                            self.distance = self.distance + line.distance
                        else:
                            self.path.append(temp)
                            self.distance = self.distance + line.distance
                        break
        else:
            self.end = True
        return temp2



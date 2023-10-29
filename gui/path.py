from vector2d import Vector2D
from scat import Scat
from copy import copy
from line import Line
from math import sqrt



class Path(object):

    def __init__(self, path = [], scat = 0, arrive = False, time = 0):
        self.path = path
        path.append(scat)
        #True if the path arrives at the destination
        self.arrive = arrive
        #The name distance is a hangover from before time was calculated. It is calculating the total time of the path
        self.distance = time

    #Make Paths is to create unoptimal paths after the best path is found
    def makePaths(self, extraPaths = 0, destination = 0, scats=[],xM=0):
        #60 km/h converted to km/s
        speed = 1/60
        temp = []
        j = 0
        #It goes through each node, starting at Origin to see if there's a neighbour it doesn't visit
        #If there is, it creates a new path that visits it from the current node. This continues until extraPaths is 0
        for node in self.path:
            i = 0
            #Finding the object scat that matches the ID node
            for scat in scats:
                if (node == scat.SCAT):
                    break
                i += 1
            #Check if at end of path
            if (node != destination):
                for neighbour in scats[i].neighbours:
                    #Break when no more paths to make
                    if (extraPaths == 0):
                        break
                    l = 0
                    #Finding the object scat that matches the ID neighbour
                    for scat in scats:
                        if (neighbour == scat.SCAT):
                            break
                        l += 1
                    if (j==0):
                        #Check if neighbour is not the destination/origin or directly ahead
                        if (neighbour != destination and neighbour != self.path[0] and neighbour != self.path[j+1]):
                            tempAr = [self.path[0]]
                            m = 0
                            for scat in scats:
                                if (self.path[0] == scat.SCAT):
                                    break
                                m += 1
                            #Create new path
                            #There's a lot to unpack here. It passes on the path up to this point and the neighbour that the path will expand next. 
                            #The time calculation takes the length of the line via Pythagoras, then converts that to actual meters using the ratio calculated in world, xM
                            #And then, because flow rate was never implemented, its divided by the max speed of the vehicle in m/s to get time taken. 
                            #As this is the first node it's assumed that they didn't need to wait at the intersection for 30s and could drive immediately
                            temp.append(Path(copy(tempAr), copy(int(neighbour)), False, copy(sqrt((scats[l].pos.x - scats[m].pos.x)*(scats[l].pos.x - scats[m].pos.x)+(scats[l].pos.y - scats[m].pos.y)*(scats[l].pos.y - scats[m].pos.y))*xM/speed)))
                            extraPaths -= 1
                    #Check if neighbour is not the destination/origin or directly ahead/behind
                    elif (neighbour != destination and neighbour != self.path[0] and neighbour != self.path[j-1] and neighbour != self.path[j+1]):
                        k = 0
                        tempAr = []
                        #As mentioned previously, it's assumed to not wait at the starting intersection for 30s, and this is to correct for adding 30s later
                        tempDis = -30
                        #Create a path identical to its own up to the node it's checking
                        while (k < j+1):
                            tempAr.append(self.path[k])
                            #Getting the time it takes to travel to the node it's checking
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
                                tempDis += (30 + sqrt((scats[m].pos.x - scats[n].pos.x)*(scats[m].pos.x - scats[n].pos.x)+(scats[m].pos.y - scats[n].pos.y)*(scats[m].pos.y - scats[n].pos.y))*xM/speed)
                            k += 1
                        m = 0
                        for scat in scats:
                            if (self.path[k] == scat.SCAT):
                                break
                            m += 1
                        #Create new path
                        temp.append(Path(copy(tempAr), copy(int(neighbour)), False, copy(tempDis + 30 + sqrt((scats[l].pos.x - scats[m].pos.x)*(scats[l].pos.x - scats[m].pos.x)+(scats[l].pos.y - scats[m].pos.y)*(scats[l].pos.y - scats[m].pos.y))*xM/speed)))
                        extraPaths -= 1
            j += 1
        return temp
    
    #Search finds the most optimal path given the path it already took and without backtracking
    #Without backtracking is important as otherwise there would be paths that jut out in a random direction and then u-turn back
    def search(self, destination = 0, scats=[],xM=0):
        #60 km/h converted to m/s
        speed = 1/60
        #This is resetting the scat objects, as search stores data on them
        for scat in scats:
            scat.fSCAT = float(0000)
            scat.distance = 0
            #Sets current to the latest node on the path
            if (float(self.path[len(self.path)-1]) == scat.SCAT):
                current = scat
        searchList = []
        searching = True
        #Search loop that uses Dijkstra's algorithm
        while(searching):
            for neighbour in current.neighbours:
                #Resets scat data. Probably could've been outside the search loop
                visited = False
                for scat in self.path:
                    #Check to avoid backtracking
                    if (scat == neighbour):
                        visited = True
                        break
                #This is to initialise nbr (neighbour) outside of the for loop
                nbr = current
                for scat in scats:
                    if (neighbour == scat.SCAT):
                        nbr = scat
                        break
                #If a neighbour hasn't been visited, store how long it would take to get there
                if (visited == False):
                    tempDist = current.distance + 30 + copy(sqrt((current.pos.x - nbr.pos.x)*(current.pos.x - nbr.pos.x)+(current.pos.y - nbr.pos.y)*(current.pos.y - nbr.pos.y))*xM/speed)
                    #Check to see if it's not on the searchlist
                    if (nbr.distance == 0):
                        nbr.fSCAT = copy(current.SCAT)
                        nbr.distance = copy(tempDist)
                        searchList.append(nbr)
                    #If it is on the search list, if going from this node is faster, overwrite the data
                    elif (nbr.distance > tempDist):
                        nbr.fSCAT = copy(current.SCAT)
                        nbr.distance = copy(tempDist)
            #If there are no more nodes to search, stop searching. This means that Destination hasn't been reach
            if (len(searchList) == 0):
                break
            #Sort the list by fastest time first
            searchList.sort(key=lambda x: x.distance, reverse=False)
            current = searchList.pop(0)
            #If the next node to search is the Destination, end search, arrive at destination is true and record the time it took
            if (current.SCAT == destination):
                searching = False
                self.arrive = True
                self.distance += current.distance
                tempPath = []
                tempSearch = True
                #Temp search is to get the data stored on scat to stitch together the path.
                #This works because the scat holds the node that comes before it in the optimal path
                while (tempSearch):
                    tempPath.insert(0,current.SCAT)
                    for scat in scats:
                        if (current.fSCAT == scat.SCAT):
                            current = scat
                            break
                    #Check to see if the temp path is finished and can be added to the actual path
                    if (float(current.SCAT) == float(self.path[len(self.path)-1])):
                        tempSearch = False
                        self.path = self.path + tempPath
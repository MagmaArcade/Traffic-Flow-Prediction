'''A 2d world that supports agents with steering behaviour

Created for COS30002 AI for Games by Clinton Woodward <cwoodward@swin.edu.au>

For class use only. Do not publically share or post this code without permission.

'''

from vector2d import Vector2D
from matrix33 import Matrix33
from line import Line
from scat import Scat
from graphics import egi
import pandas as pd
from data import process_data



class World(object):

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.df = process_data('Scats Data October 2006.csv')
        self.mix = self.df.loc[0, 'NB_LONGITUDE']
        self.max = self.df.loc[0, 'NB_LONGITUDE']
        self.miy = self.df.loc[0, 'NB_LATITUDE']
        self.may = self.df.loc[0, 'NB_LATITUDE']
        self.latM = 111.11
        self.xM = 0
        self.scats = []
        self.origin = 3001
        self.destination = 2846
        self.successes = []
        self.successInt = 0
        self.toggle = False
    
        i = 1
        while (i<self.df.shape[0]):
            if (self.df.loc[i, 'NB_LONGITUDE'] < self.mix):
                self.mix = self.df.loc[i, 'NB_LONGITUDE']
            if (self.max < self.df.loc[i, 'NB_LONGITUDE']):
                self.max = self.df.loc[i, 'NB_LONGITUDE']
            if (self.miy > self.df.loc[i, 'NB_LATITUDE']):
                self.miy = self.df.loc[i, 'NB_LATITUDE']
            if (self.may < self.df.loc[i, 'NB_LATITUDE']):
                self.may = self.df.loc[i, 'NB_LATITUDE']
            i +=1
        
        self.xM = (self.max - self.mix)*self.latM/(self.cx*0.625)
        self.ratio = (self.max-self.mix)/(self.may - self.miy)



        i = 0
        self.my = self.may-self.miy
        self.mx = self.max-self.mix
        while (i<self.df.shape[0]):
            self.scats.append(Scat(self,Vector2D((self.df.loc[i,'NB_LONGITUDE']-self.mix)/self.mx*self.cx*0.675 +self.cx*0.05, (self.df.loc[i,'NB_LATITUDE']-self.miy)/self.ratio/self.my*self.cy*0.9 +self.cy*0.05),self.df.loc[i,'SCATS_Number'],self.df.loc[i,'SCATS Neighbours'].split(" ")))
            i +=1

        i = 0
        self.lines = []
        while (i<self.df.shape[0]):
            x = self.df.loc[i,'SCATS Neighbours'].split(" ")
            for j in x:
                if (float(j) > self.df.loc[i,'SCATS_Number']):
                    k = i
                    while (k<self.df.shape[0]):
                        if (float(j) == self.df.loc[k,'SCATS_Number']):
                            temp = (),
                            self.lines.append(Line(self,Vector2D((self.df.loc[i,'NB_LONGITUDE']-self.mix)/self.mx*self.cx*0.675 +self.cx*0.05, (self.df.loc[i,'NB_LATITUDE']-self.miy)/self.ratio/self.my*self.cy*0.9 +self.cy*0.05),Vector2D((self.df.loc[k,'NB_LONGITUDE']-self.mix)/self.mx*self.cx*0.675 +self.cx*0.05, (self.df.loc[k,'NB_LATITUDE']-self.miy)/self.ratio/self.my*self.cy*0.9 +self.cy*0.05),self.df.loc[i,'SCATS_Number'],self.df.loc[k,'SCATS_Number']))
                        k+=1
            i += 1

        for scat in self.scats:
            if (scat.SCAT == self.origin):
                scat.color = "ORANGE"
            elif (scat.SCAT == self.destination):
                scat.color = "BLUE"

        

    def reset(self):
        for line in self.lines:
            line.color = "RED"
            self.successes = []
            self.successInt = -1
    
    def resetOrigin(self):
        for scat in self.scats:
            if (scat.SCAT == self.origin):
                scat.color = "GREEN"
                break

    def resetDestination(self):
        for scat in self.scats:
            if (scat.SCAT == self.destination):
                scat.color = "GREEN"
                break

    def switchRoute(self):
        for line in self.lines:
            line.color = "RED"
        if ((self.successInt +1) < len(self.successes)):
            self.successInt = self.successInt +1
        else:
            self.successInt = 0
        i =1
        while (i < len(self.successes[self.successInt].path)):
            for line in self.lines:
                if ((line.SCAT1 == self.successes[self.successInt].path[i-1] or line.SCAT1 == self.successes[self.successInt].path[i]) and (line.SCAT2 == self.successes[self.successInt].path[i-1] or line.SCAT2 == self.successes[self.successInt].path[i])):
                    line.color = "ORANGE"
            i += 1

        
        

    def render(self):

        for line in self.lines:
            line.render()
            
        for scat in self.scats:
            scat.render()

        egi.color = "WHITE"
        egi.text_at_pos(0.8*self.cx, 0.9*self.cy, "Origin: " + str(self.origin))
        egi.text_at_pos(0.8*self.cx, 0.85*self.cy, "Destination: " + str(self.destination))
        if (self.toggle == True):
            egi.text_at_pos(0.8*self.cx, 0.8*self.cy, "Cursor: Destination")
        if (self.toggle == False):
            egi.text_at_pos(0.8*self.cx, 0.8*self.cy, "Cursor: Origin")
        if (len(self.successes) >0):
            temp = int(self.successes[self.successInt].distance/60) - self.successes[self.successInt].distance/60
            if (temp < 0):
                temp = temp * -1
            else:
                temp = 1-temp

            egi.text_at_pos(0.8*self.cx, 0.75*self.cy, "Time Taken: " + str(int(self.successes[self.successInt].distance/60)) +"m " + str(int(temp*60)) + "s")
    


    def transform_point(self, point, pos, forward, side):
        # make a copy of original points (so we don't trash them)
        wld_pt = point.copy()
        # create a transformation matrix to perform the operations
        mat = Matrix33()
        # rotate
        mat.rotate_by_vectors_update(forward, side)
        # and translate
        mat.translate_update(pos.x, pos.y)
        # now transform all the points (vertices)
        mat.transform_vector2d(wld_pt)
        # done
        return wld_pt

    def transform_points(self, points, pos, forward, side, scale):
        ''' Transform the given list of points, using the provided position,
            direction and scale, to object world space. '''
        # make a copy of original points (so we don't trash them)
        wld_pts = [pt.copy() for pt in points]
        # create a transformation matrix to perform the operations
        mat = Matrix33()
        # scale,
        mat.scale_update(scale.x, scale.y)
        # rotate
        mat.rotate_by_vectors_update(forward, side)
        # and translate
        mat.translate_update(pos.x, pos.y)
        # now transform all the points (vertices)
        mat.transform_vector2d_list(wld_pts)
        # done
        return wld_pts

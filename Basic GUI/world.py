'''A 2d world that supports agents with steering behaviour

Created for COS30002 AI for Games by Clinton Woodward <cwoodward@swin.edu.au>

For class use only. Do not publically share or post this code without permission.

'''

from vector2d import Vector2D
from matrix33 import Matrix33
from graphics import egi
import pandas as pd
from data import process_data



class World(object):

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.target = Vector2D(cx / 4, 3*cy / 4)
        self.target2 = Vector2D(cx / 4, cy / 4)
        self.hunter = None
        self.agents = []
        self.bullets = []
        self.mtargets = []
        self.paused = True
        self.show_info = True
        self.targ = 1
        self.df = process_data('Scats Data October 2006.csv')
        self.mix = self.df.loc[0, 'NB_LONGITUDE']
        self.max = self.df.loc[0, 'NB_LONGITUDE']
        self.miy = self.df.loc[0, 'NB_LATITUDE']
        self.may = self.df.loc[0, 'NB_LATITUDE']
        self.dots = []
    
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
        i = 0
        self.my = self.may-self.miy
        self.mx = self.max-self.mix
        while (i<self.df.shape[0]):
            self.dots.append(Vector2D((self.df.loc[i,'NB_LONGITUDE']-self.mix)/self.mx*self.cx*0.9 +self.cx*0.05, (self.df.loc[i,'NB_LATITUDE']-self.miy)/self.my*self.cy*0.9 +self.cy*0.05))
            i +=1

        i = 0
        self.lines = []
        while (i<self.df.shape[0]):
            x = self.df.loc[i,'SCATS Neighbours'].split()
            for j in x:
                if (int(j) > self.df.loc[i,'SCATS_Number']):
                    k = i
                    while (k<self.df.shape[0]):
                        if (int(j) == self.df.loc[k,'SCATS_Number']):
                            temp = (Vector2D((self.df.loc[i,'NB_LONGITUDE']-self.mix)/self.mx*self.cx*0.9 +self.cx*0.05, (self.df.loc[i,'NB_LATITUDE']-self.miy)/self.my*self.cy*0.9 +self.cy*0.05)),Vector2D((self.df.loc[k,'NB_LONGITUDE']-self.mix)/self.mx*self.cx*0.9 +self.cx*0.05, (self.df.loc[k,'NB_LATITUDE']-self.miy)/self.my*self.cy*0.9 +self.cy*0.05)
                            self.lines.append(temp)
                        k+=1
                    

            print(self.lines)
            i += 1




    #def update(self, delta):
    #    if not self.paused:
            #for agent in self.agents:
            #    agent.update(delta)

    #        for bullet in self.bullets:
    #            bullet.update(delta)

    #        for targ in self.mtargets:
    #            targ.update(delta)

    def render(self):

        
        for agent in self.agents:
            agent.render()

        for bullet in self.bullets:
            bullet.render()

        for targ in self.mtargets:
            targ.render()

        egi.red_pen()
        for j in self.lines:
            egi.line_by_pos(j[0],j[1])

        egi.green_pen()
        i=0
        while (i<self.df.shape[0]):
            egi.cross(self.dots[i], 10)
            i +=1
            
        

        #if self.target:
        #    egi.red_pen()
        #    egi.cross(self.target, 10)

        #if self.target2:
        #    egi.blue_pen()
        #    egi.cross(self.target2, 10)

        #if self.show_info:
            #infotext = ', '.join(set(agent.mode for agent in self.agents))
            #egi.white_pen()
            #egi.text_at_pos(0, 0, infotext)

    def wrap_around(self, pos):
        ''' Treat world as a toroidal space. Updates parameter object pos '''
        max_x, max_y = self.cx, self.cy
        if pos.x > max_x:
            pos.x = pos.x - max_x
        elif pos.x < 0:
            pos.x = max_x - pos.x
        if pos.y > max_y:
            pos.y = pos.y - max_y
        elif pos.y < 0:
            pos.y = max_y - pos.y

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

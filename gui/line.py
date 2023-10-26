from vector2d import Vector2D
from graphics import egi
from math import sqrt

class Line(object):

    def __init__(self, world=None, pos1 = Vector2D, pos2 = Vector2D, SCAT1 = 0, SCAT2 = 0):
        self.world = world
        self.pos1 = pos1
        self.pos2 = pos2
        self.SCAT1 = float(SCAT1)
        self.SCAT2 = float(SCAT2)
        self.distance = sqrt((self.pos1.x - self.pos2.x)*(self.pos1.x - self.pos2.x)+(self.pos1.y - self.pos2.y)*(self.pos1.y - self.pos2.y))
        self.color = 'RED'

    def render(self, color=None):
        egi.set_pen_color(name=self.color)
        egi.line_by_pos(self.pos1, self.pos2)
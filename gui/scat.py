from vector2d import Vector2D
from graphics import egi



class Scat(object):

    def __init__(self, world=None, pos = Vector2D, SCAT = 0, neighbours = []):
        self.world = world
        self.pos = pos
        self.SCAT = SCAT
        self.neighbours = []
        self.color = 'GREEN'

        self.distance = 0
        self.fSCAT = float(0000)

        for string in neighbours:
            self.neighbours.append(float(string))

    def render(self, color=None):
        egi.set_pen_color(name=self.color)
        egi.cross(self.pos, 10)
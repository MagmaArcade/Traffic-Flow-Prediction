from vector2d import Vector2D
from graphics import egi
import pandas as pd
from math import sqrt



class Scat(object):

    def __init__(self, world=None, pos = Vector2D, SCAT = 0, neighbours = []):
        self.world = world
        self.pos = pos
        self.SCAT = SCAT
        self.neighbours = []
        self.color = 'GREEN'
        self.exitLa = []
        self.exitLo = []

        #Data storage for the search loop in path.py
        self.distance = 0
        self.fSCAT = float(0000)

        for string in neighbours:
            self.neighbours.append(float(string))
        
        # Read data file
        filtered_df = pd.read_csv("Scats Data.csv", encoding='utf-8').fillna(0)

        # Filter the DataFrame around the SCATS, making a new DataFrame that just has this SCAT
        filtered_df = filtered_df[filtered_df['SCATS_Number'] == int(SCAT)]

        # Remove duplicates if there are any
        filtered_df = filtered_df.drop_duplicates(subset=['NB_LATITUDE', 'NB_LONGITUDE'])
        filtered_df = filtered_df.reset_index()

        i = 0
        while (i + 1 < len(filtered_df)):
            self.exitLa.append(float(filtered_df.loc[i,'NB_LATITUDE']))
            self.exitLo.append(float(filtered_df.loc[i,'NB_LONGITUDE']))
            i += 1


    def findClosest(self, scat):
        i = 1
        distance = sqrt(abs(self.exitLa[0]-scat.exitLa[0])*abs(self.exitLa[0]-scat.exitLa[0])+abs(self.exitLo[0]-scat.exitLo[0])*abs(self.exitLa[0]-scat.exitLo[0]))
        bestI = 0
        while i < len(self.exitLa):
            distance2 = sqrt(abs(self.exitLa[i]-scat.exitLa[0])*abs(self.exitLa[i]-scat.exitLa[0])+abs(self.exitLo[i]-scat.exitLo[0])*abs(self.exitLo[i]-scat.exitLo[0]))
            if ( distance2 < distance):
                distance = sqrt(abs(self.exitLa[i]-scat.exitLo[0])*abs(self.exitLa[i]-scat.exitLa[0])+abs(self.exitLo[i]-scat.exitLo[0])*abs(self.exitLo[i]-scat.exitLo[0]))
                bestI = i
            i += 1


        return self.exitLa[bestI], self.exitLo[bestI]



    def render(self, color=None):
        egi.set_pen_color(name=self.color)
        egi.cross(self.pos, 10)
from vector2d import Vector2D
from line import Line
from scat import Scat
from graphics import egi
import pandas as pd


class World(object):
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        
        self.model="lstm"

        # Read in CSV file
        df1 = pd.read_csv("Scats Data.csv", encoding='utf-8').fillna(0) 

        i = df1.shape[0] -1
        pr = 0000
        #Shortens the DF to remove duplicates
        while (i > -1):
            cr = df1.loc[i, 'SCATS_Number']
            if (cr == pr):
                df1 = df1.drop([i])
            pr = cr
            i -=1
        
        #Get data via data.py
        self.df = df1.reset_index(drop=True)
        #These are used to scale the lines/nodes appropriately
        self.mix = self.df.loc[0, 'NB_LONGITUDE']
        self.max = self.df.loc[0, 'NB_LONGITUDE']
        self.miy = self.df.loc[0, 'NB_LATITUDE']
        self.may = self.df.loc[0, 'NB_LATITUDE']
        #1 degree = 111.11km
        self.latM = 111.11
        self.scats = []
        #Arbitrarily decided Origin and Destination
        self.origin = 3001
        self.destination = 2846
        self.successes = []
        #Used for cycling through showing successful paths
        self.successInt = 0
        self.toggle = False

        #Finding the largest/smallest values for Lat/long
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
        self.my = self.may-self.miy
        self.mx = self.max-self.mix
        #The distance between the furhest points latitude from each other, converted to km and divided by the max x length
        #which equals how many km 1 x unit equals
        self.xM = (self.mx)*self.latM/(self.cx*0.625)
        #The ratio of how long a distance the nodes are vs how tall
        self.ratio = (self.mx)/(self.my)



        i = 0
        #Create scat objects with x y positions scaled based on their longitude/latitude
        while (i<self.df.shape[0]):
            self.scats.append(Scat(self,Vector2D((self.df.loc[i,'NB_LONGITUDE']-self.mix)/self.mx*self.cx*0.675 +self.cx*0.05, (self.df.loc[i,'NB_LATITUDE']-self.miy)/self.ratio/self.my*self.cy*0.9 +self.cy*0.05),self.df.loc[i,'SCATS_Number'],self.df.loc[i,'SCATS Neighbours'].split(" ")))
            i +=1

        #Create lines based on the neighbours of the scat objects
        i = 0
        self.lines = []
        while (i<self.df.shape[0]):
            x = self.df.loc[i,'SCATS Neighbours'].split(" ")
            for j in x:
                #Check to only create lines for neighbours with a higher SCATS number. This is to avoid creating duplicate lines
                if (float(j) > self.df.loc[i,'SCATS_Number']):
                    k = i
                    while (k<self.df.shape[0]):
                        if (float(j) == self.df.loc[k,'SCATS_Number']):
                            temp = (),
                            self.lines.append(Line(self,Vector2D((self.df.loc[i,'NB_LONGITUDE']-self.mix)/self.mx*self.cx*0.675 +self.cx*0.05, (self.df.loc[i,'NB_LATITUDE']-self.miy)/self.ratio/self.my*self.cy*0.9 +self.cy*0.05),Vector2D((self.df.loc[k,'NB_LONGITUDE']-self.mix)/self.mx*self.cx*0.675 +self.cx*0.05, (self.df.loc[k,'NB_LATITUDE']-self.miy)/self.ratio/self.my*self.cy*0.9 +self.cy*0.05),self.df.loc[i,'SCATS_Number'],self.df.loc[k,'SCATS_Number']))
                        k+=1
            i += 1

        #Assign colours to the scat objects that are origin and destination
        for scat in self.scats:
            if (scat.SCAT == self.origin):
                scat.color = "ORANGE"
            elif (scat.SCAT == self.destination):
                scat.color = "BLUE"

        
    #Resets lines back to default colour
    def reset(self):
        for line in self.lines:
            line.color = "RED"
            self.successes = []
            self.successInt = -1
    
    #Reset origin colour back to default
    def resetOrigin(self):
        for scat in self.scats:
            if (scat.SCAT == self.origin):
                scat.color = "GREEN"
                break

    #Reset destination colour back to default
    def resetDestination(self):
        for scat in self.scats:
            if (scat.SCAT == self.destination):
                scat.color = "GREEN"
                break

    def recolourDO(self):
        for scat in self.scats:
            if (scat.SCAT == self.origin):
                scat.color = "ORANGE"
            elif (scat.SCAT == self.destination):
                scat.color = "BLUE"

    #Cycle through displaying successful routes
    def switchRoute(self):
        #Reset line colours back to default
        for line in self.lines:
            line.color = "RED"
        #If there is another route to cycle through than go to it
        if ((self.successInt +1) < len(self.successes)):
            self.successInt = self.successInt +1
        #Otherwise reset route
        else:
            self.successInt = 0
        i =1
        #Make currently displayed route lines to be orange
        while (i < len(self.successes[self.successInt].path)):
            for line in self.lines:
                if ((line.SCAT1 == self.successes[self.successInt].path[i-1] or line.SCAT1 == self.successes[self.successInt].path[i]) and (line.SCAT2 == self.successes[self.successInt].path[i-1] or line.SCAT2 == self.successes[self.successInt].path[i])):
                    line.color = "ORANGE"
            i += 1
        
    #Draw all visible objects onto screen
    def render(self):

        for line in self.lines:
            line.render()
            
        for scat in self.scats:
            scat.render()

        egi.color = "WHITE"

      # Command Notation printed onto the map 
        commands = [
        "Tab: Toggle Origin/Dest",
        "Space: Calculate Path",
        "Q: Toggle Top 5 Paths",
        "L_Click: Select Node",
        "COMMANDS"
        ]

        for i, command in enumerate(commands):
            egi.text_at_pos(0.75*self.cx, 0.75*self.cy + i*30, command)

        #Render text onto screen
        egi.text_at_pos(0.75*self.cx, 0.65*self.cy, "Origin: " + str(self.origin))
        egi.text_at_pos(0.75*self.cx, 0.6*self.cy, "Destination: " + str(self.destination))
        if (self.toggle == True):
            egi.text_at_pos(0.75*self.cx, 0.55*self.cy, "Cursor: Destination")
        if (self.toggle == False):
            egi.text_at_pos(0.75*self.cx, 0.55*self.cy, "Cursor: Origin")
        if (self.model == "lstm"):
            egi.text_at_pos(0.75*self.cx, 0.45*self.cy, "Model: Lstm")
        elif (self.model == "gru"):
            egi.text_at_pos(0.75*self.cx, 0.45*self.cy, "Model: Gru")
        elif (self.model == "saes"):
            egi.text_at_pos(0.75*self.cx, 0.45*self.cy, "Model: Saes")
        if (len(self.successes) >0):
            temp = int(self.successes[self.successInt].distance/60) - self.successes[self.successInt].distance/60
            if (temp < 0):
                temp = temp * -1
            else:
                temp = 1-temp

            egi.text_at_pos(0.75*self.cx, 0.5*self.cy, "Time Taken: " + str(int(self.successes[self.successInt].distance/60)) +"m " + str(int(temp*60)) + "s")
            #egi.text_at_pos(0.8*self.cx, 0.75*self.cy, "Current Path: " + )





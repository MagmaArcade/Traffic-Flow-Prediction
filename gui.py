from graphics import egi, KEY
from pyglet import window, clock
from pyglet.gl import *
from vector2d import Vector2D
from world import World
from path import Path
from math import sqrt
import argparse



def on_key_press(symbol, modifiers):
    #Calculates up to 5 routes from Origin to Destination
    if (symbol== KEY.SPACE):
        #Resets line colouring
        world.reset()
        #Possible amount of additional routes from Origin to Destination
        extraPaths = 4
        paths = [Path([],world.origin,False,0, args.time, world.model)]

        #Optimal path completes its search
        paths[0].search(world.destination,world.scats,world.xM)
        #Optimal path data used to make the suboptimal path beginings
        paths = paths + paths[0].makePaths(extraPaths,world.destination, world.scats,world.xM)
        i = 0
        for path in paths:
            #If check so only suboptimal paths are searched
            if (i>0):
                path.search(world.destination,world.scats,world.xM)
                if (path.arrive == True):
                    world.successes.append(path)
            #If check so that if optimal path fails to get to Destination, its not auto added to the successes
            if (i == 0 and path.arrive == True):
                world.successes.append(path)
            i += 1
        #Order successful paths by time
        world.successes.sort(key=lambda x: x.distance, reverse=False)
        #Display new routes
        world.switchRoute()

    #Switch currently displayed route from Origin to Destination
    elif (symbol == KEY.Q):
        world.switchRoute()

    #Origin/Destination toggle
    elif (symbol == KEY.TAB):
        if (world.toggle):    
            world.toggle = False
        else:
            world.toggle = True
    
    elif (symbol == KEY.W):
        if (world.model =='saes'):    
            world.model = 'lstm'
        elif (world.model =='lstm'):
            world.model = 'gru'
        else:
            world.model = 'saes'

    
def on_mouse_press(x, y, button, modifiers):
    #Couldn't get right click to work, so changing Origin/Destination is decided by a toggle
    if (button == 1 and world.toggle == False):
        for scat in world.scats:
            if (sqrt((scat.pos.x-x)*(scat.pos.x-x)+(scat.pos.y-y)*(scat.pos.y-y)) < 10):
                world.resetOrigin()
                world.origin = scat.SCAT
                scat.color = "ORANGE"

    elif (button == 1 and world.toggle):
        for scat in world.scats:
            if (sqrt((scat.pos.x-x)*(scat.pos.x-x)+(scat.pos.y-y)*(scat.pos.y-y)) < 10):
                world.resetDestination()
                world.destination = scat.SCAT
                scat.color = "BLUE"


def on_resize(cx, cy):
    world.cx = cx
    world.cy = cy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin",
        default=3662,
        help="SCATS site number.")
    parser.add_argument(
        "--destination",
        default=2846,
        help="SCATS site number.")
    parser.add_argument(
        "--time",
        default="9:30",
        help="The time")
    parser.add_argument(
        "--date",
        default="1/10/06",
        help="The day of the month")
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to use for prediction (lstm, gru, saes)")
    args = parser.parse_args()


    # create a pyglet window and set glOptions
    win = window.Window(width=850, height=650, vsync=True, resizable=True)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # needed so that egi knows where to draw
    egi.InitWithPyglet(win)
    # prep the fps display
    fps_display = window.FPSDisplay(win)
    # register key and mouse event handlers
    win.push_handlers(on_key_press)
    win.push_handlers(on_mouse_press)
    win.push_handlers(on_resize)

    # create a world for agents
    world = World(850, 650)
    world.resetOrigin()
    world.resetDestination()
    world.origin =args.origin
    world.destination = args.destination
    world.recolourDO()
   

    while not win.has_exit:
        win.dispatch_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # show nice FPS bottom right (default)
        delta = clock.tick()
        #world.update(delta)
        world.render()
        #fps_display.draw()
        # swap the double buffer
        win.flip()

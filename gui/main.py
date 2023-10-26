from graphics import egi, KEY
from pyglet import window, clock
from pyglet.gl import *
from vector2d import Vector2D
from world import World
from path import Path
from math import sqrt

def on_key_press(symbol, modifiers):
    if (symbol== KEY.SPACE):
        world.reset()
        extraPaths = 4
        paths = [Path([],world.origin,False,0)]


        paths[0].search(world.destination,world.scats,world.xM)
        paths = paths + paths[0].makePaths(extraPaths,world.destination, world.scats,world.xM)
        i = 0
        for path in paths:
            if (i>0):
                path.search(world.destination,world.scats,world.xM)
                if (path.arrive == True):
                    world.successes.append(path)
            if (i == 0 and path.arrive == True):
                world.successes.append(path)
            i += 1
        world.successes.sort(key=lambda x: x.distance, reverse=False)
        world.switchRoute()

    elif (symbol == KEY.Q):
        world.switchRoute()

    elif (symbol == KEY.TAB):
        if (world.toggle):    
            world.toggle = False
        else:
            world.toggle = True

    
def on_mouse_press(x, y, button, modifiers):

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


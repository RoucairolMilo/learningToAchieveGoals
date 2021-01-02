import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

import numpy as np
class dirtyPlot :

    text = {}


    def __init__(selfs, gridSize) :
        global text
        global im
        fig, ax = plt.subplots()
        cmap = mcolors.ListedColormap(['white', 'blue', 'yellow', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        gridData = np.random.rand(gridSize[0], gridSize[1]) * 2 - 0.5
        im = ax.imshow(gridData, cmap=cmap, norm=norm)

        grid = np.arange(-0.5, 11, 1)
        xmin, xmax, ymin, ymax = -0.5, 10.5, -0.5, 10.5
        lines = ([[(x, y) for y in (ymin, ymax)] for x in grid]
                 + [[(x, y) for x in (xmin, xmax)] for y in grid])
        grid = mcoll.LineCollection(lines, linestyles='solid', linewidths=2,
                                    color='teal')
        ax.add_collection(grid)

        text = {}

        plt.ion()
        plt.show()

    def update(self,gridSize, state, rewardPos, choice, DGTable, Qdict, rew, method, rounding):
        global im
        gridData = np.zeros((gridSize[0], gridSize[1]))
        gridData[state[0]][state[1]] = 1
        gridData[rewardPos[0]][rewardPos[1]] = 3 if rew else 2
        im.set_data(gridData)

        decaly = 0
        decalx = 0

        if(choice == "haut") :
          decaly = 0.25
        if (choice == "bas"):
          decaly = -0.25
        if (choice == "gauche"):
          decalx = -0.25
        if (choice == "droite"):
          decalx = 0.25
        """        
        if text.get((state[0], state[1], choice), 0) != 0:
          text.get((state[0], state[1], choice), 0).set_visible(False)
        if method == 'DG' :
          tex = plt.text(state[1] + decaly - 0.2, state[0] + decalx + 0.2, str(round(DGTable.get((state, choice, rewardPos), 0), rounding)), color="red", fontsize=8)
          text[(state[0], state[1], choice)] = tex
        if method == 'Q' :
          tex = plt.text(state[1] + decaly - 0.2, state[0] + decalx + 0.2, str(round(Qdict.get((state, choice), 0), rounding)), color="red", fontsize=8)
          text[(state[0], state[1], choice)] = tex
        """

        if text.get((state[0], state[1], choice), 0) != 0:
            text.get((state[0], state[1], choice), 0).set_text(str(round(DGTable.get((state, choice, rewardPos), 0), rounding)))
        else :
            if method == 'DG':
                tex = plt.text(state[1] + decaly - 0.2, state[0] + decalx + 0.2,
                               str(round(DGTable.get((state, choice, rewardPos), 0), rounding)), color="red",
                               fontsize=8)
                text[(state[0], state[1], choice)] = tex
            if method == 'Q':
                tex = plt.text(state[1] + decaly - 0.2, state[0] + decalx + 0.2,
                               str(round(Qdict.get((state, choice), 0), rounding)), color="red", fontsize=8)
                text[(state[0], state[1], choice)] = tex

        plt.draw()
        plt.pause(0.01)
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def showPlot(points):
    plt.switch_backend('agg')
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
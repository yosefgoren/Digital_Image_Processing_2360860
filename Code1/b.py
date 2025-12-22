#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation
import click

def loadbob():
    mat = loadmat("instructions/SpongeBob.mat")
    rawbob = mat["SpongeBob"]
    assert rawbob.ndim == 3

    return np.moveaxis(rawbob, -1, 0)   # (Hight, Width, Time) -> (Time, Hight, Width)

def show_ani(ani_mat, interval: int):
    T, H, W = ani_mat.shape

    fig, ax = plt.subplots()
    # Use global min/max so contrast is not determined by the (zero) first frame
    vmin, vmax = ani_mat.min(), ani_mat.max()
    im = ax.imshow(ani_mat[0], cmap="gray", animated=True, vmin=vmin, vmax=vmax)
    ax.axis("off")

    def update(t):
        print(t)
        im.set_array(ani_mat[t])
        return (im,)

    ani = FuncAnimation(fig, update, frames=T, interval=interval)
    plt.show()


@click.command("showbob")
@click.argument("interval", type=int, default=20)
@click.argument("diff_first", type=bool, default=False)
def showbob(interval: int, diff_first: bool):
    ani_mat = loadbob()
    
    if diff_first:
        ani_mat = ani_mat - ani_mat[0]
    
    show_ani(ani_mat, interval)

if __name__ == "__main__":
    showbob()
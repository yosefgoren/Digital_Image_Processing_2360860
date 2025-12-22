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

def show_ani(ani_mat, interval: int, manual: bool = False):
    T, H, W = ani_mat.shape

    fig, ax = plt.subplots()
    # Use global min/max so contrast is not determined by the (zero) first frame
    vmin, vmax = ani_mat.min(), ani_mat.max()
    im = ax.imshow(ani_mat[0], cmap="gray", animated=True, vmin=vmin, vmax=vmax)
    ax.axis("off")

    if manual:
        # Manual mode: use keyboard to step through frames.
        # Keys:
        # - Right arrow / 'n' : next frame
        # - Left arrow  / 'p' : previous frame
        # - 'q'               : quit/close window
        state = {"t": 0}

        def on_key(event):
            if event.key in ("right", "n"):
                state["t"] = (state["t"] + 1) % T
            elif event.key in ("left", "p"):
                state["t"] = (state["t"] - 1) % T
            elif event.key == "q":
                plt.close(fig)
                return
            
            next_frame_index = state["t"]
            print(f"Show frame number: {next_frame_index}")
            im.set_array(ani_mat[next_frame_index])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)
    else:
        def update(next_frame_index: int):
            print(f"Showing frame number: {next_frame_index}")
            im.set_array(ani_mat[next_frame_index])
            return (im,)

        anim = FuncAnimation(fig, update, frames=T, interval=interval)
    plt.show()


@click.command("showbob")
@click.argument("interval", type=int, default=20)
@click.argument("diff_first", type=bool, default=False)
@click.option("--manual", is_flag=True, help="Step through frames manually with keyboard.")
def showbob(interval: int, diff_first: bool, manual: bool):
    ani_mat = loadbob()
    
    if diff_first:
        ani_mat = ani_mat - ani_mat[0]

    show_ani(ani_mat, interval, manual=manual)

if __name__ == "__main__":
    showbob()
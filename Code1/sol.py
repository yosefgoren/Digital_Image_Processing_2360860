#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass as dc
import click
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation
import math

@click.group()
class cli:
    pass

DIM_SIZE = 201


def f(x,y):
    return np.sinc(y*50/21)

@dc
class SincAnalysis:
    delta: float
    
    def idx2range(self, t):
        return (t-100)*self.delta

    def get_range_bounds(self) -> tuple:
        range_start = self.idx2range(0)
        range_end = self.idx2range(DIM_SIZE-1)

        return range_start, range_end

    def get_grid(self) -> tuple:
        grid_1d = self.idx2range(np.arange(DIM_SIZE))

        rows = grid_1d
        cols = grid_1d

        grid_2d = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1)
        print(grid_2d.shape)

        y_grid = grid_2d[..., 0]
        x_grid = grid_2d[..., 1]
        return x_grid, y_grid

    def get_img(self):
        x_grid, y_grid = self.get_grid()
        return f(x_grid, y_grid)
        
    def show_img(self):
        range_start, range_end = self.get_range_bounds()
        plt.imshow(self.get_img(), extent=(range_start, range_end, range_start, range_end))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Sinc Sampled Image")
        plt.show()

    def show_freq(self):
        img = self.get_img()

        # 2D FFT
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)

        # magnitude (log for visibility)
        mag = np.log1p(np.abs(F))

        # frequency axes
        freq = np.fft.fftshift(np.fft.fftfreq(DIM_SIZE, d=self.delta))
        fmin, fmax = freq[0], freq[-1]

        plt.imshow(
            mag,
            cmap="gray",
            extent=(fmin, fmax, fmin, fmax),
            origin="lower"
        )
        plt.xlabel("fx")
        plt.ylabel("fy")
        plt.title("Frequency domain (magnitude)")
        plt.colorbar()
        plt.show()

@cli.command("q1_b")
def q1_b():
    a = SincAnalysis(0.1)
    a.show_img()

@cli.command("q1_c")
def q1_c():
    a = SincAnalysis(0.1)
    a.show_freq()

@cli.command("q1_d")
def q1_d():
    a = SincAnalysis(1)
    a.show_img()
    a.show_freq()


def loadbob():
    mat = loadmat("instructions/SpongeBob.mat")
    rawbob = mat["SpongeBob"]
    assert rawbob.ndim == 3

    return np.moveaxis(rawbob, -1, 0)   # (Hight, Width, Time) -> (Time, Hight, Width)

MANUAL_MODE_HELP_STR = """\
Manual mode: use keyboard to step through frames.
Keys:
- Right arrow / 'n' : next frame
- Left arrow  / 'p' : previous frame
- 'q'               : quit/close window
"""

def show_bob_animation(ani_mat, display_interval: int, sampling_ratio: float = 1.0, manual: bool = False):
    T, H, W = ani_mat.shape

    fig, ax = plt.subplots()
    # Use global min/max so contrast is not determined by the (zero) first frame
    vmin, vmax = ani_mat.min(), ani_mat.max()
    im = ax.imshow(ani_mat[0], cmap="gray", animated=True, vmin=vmin, vmax=vmax)
    ax.axis("off")

    if manual:
        print(MANUAL_MODE_HELP_STR)
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
            print(f"Showing frame number: {next_frame_index}")
            im.set_array(ani_mat[next_frame_index])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)
    else:
        # In sampled auto mode, show every `sampling_ratio`-th frame and slow the refresh.
        effective_interval = display_interval * sampling_ratio
        num_steps = int(T / sampling_ratio)

        def update(step_idx: int):
            frame_idx = min(math.floor(step_idx * sampling_ratio), T - 1)
            print(f"Showing frame number: {frame_idx}")
            im.set_array(ani_mat[frame_idx])
            return (im,)

        anim = FuncAnimation(fig, update, frames=num_steps, interval=effective_interval)
    plt.show()

def showbob(
        display_interval: int = 20,
        sampling_ratio: float = 1.0,
        diff_first: bool = False,
        manual: bool = False
    ):
    ani_mat = loadbob()
    
    if diff_first:
        ani_mat = ani_mat - ani_mat[0]

    show_bob_animation(ani_mat, display_interval, sampling_ratio, manual)

@cli.command("q1_f")
def q1_f():
    showbob()

@cli.command("q1_g")
def q1_f():
    showbob(diff_first=True, manual=True)

@cli.command("q1_i")
def q1_i():
    showbob(2, 18)


if __name__ == "__main__":
    cli()
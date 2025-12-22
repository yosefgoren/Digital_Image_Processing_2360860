import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass as dc
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

for delta in [0.1, 1]:
    a = SincAnalysis(delta)
    a.show_img()
    a.show_freq()

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation

def loadbob():
    mat = loadmat("instructions/SpongeBob.mat")        # inspect keys if unsure
    rawbob = mat["SpongeBob"]                   # shape (T, H, W) or (H, W, T)
    print(rawbob.shape)

    assert vol.ndim == 3

    # ---- explicitly choose time axis ----
    # assume time is the LAST dimension (H, W, T)
    vol = np.moveaxis(vol, -1, 0)   # -> (T, H, W)

    return vol




def showbob():
    bob = loadbob()

    T, H, W = bob.shape

    fig, ax = plt.subplots()
    im = ax.imshow(bob[0], cmap="gray", animated=True)
    ax.axis("off")

    def update(t):
        im.set_array(bob[t])
        return (im,)

    ani = FuncAnimation(fig, update, frames=T, interval=20)
    plt.show()


showbob()
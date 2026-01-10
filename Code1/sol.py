#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass as dc
import click
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation
import math
from PIL import Image

@click.group()
class cli:
    pass


# =====================================================================
#                          Question 1
# =====================================================================

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

# =====================================================================
#                          Question 2
# =====================================================================

def load_image(path: str, require_rgb: bool = False) -> np.ndarray:
    img = Image.open(path)
    # Convert to RGB if necessary (handles RGBA, P, L modes)
    if img.mode != 'RGB' and require_rgb:
        img = img.convert('RGB')
    return np.array(img)

def display_images_side_by_side(img1: np.ndarray, img2: np.ndarray, titles=("Image 1", "Image 2"), fullscreen: bool = True):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if fullscreen:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

    axes[0].imshow(img1, interpolation='nearest')
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    axes[1].imshow(img2, interpolation='nearest')
    axes[1].set_title(titles[1])
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def display_image(img: np.ndarray, scale: float = 2.0, fullscreen: bool = False):

    h, w = img.shape[:2]

    # Matplotlib uses inches; default DPI is usually 100
    dpi = plt.rcParams.get('figure.dpi', 100)

    fig = plt.figure(figsize=(w * scale / dpi, h * scale / dpi), dpi=dpi)
    if fullscreen:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


def add_shifted_dimension(tensor: np.ndarray, diff: tuple) -> np.ndarray:
    """
    Add an additional dimension of size 2 to a tensor.
    The new dimension contains the original tensor at index 0 and a cyclically
    shifted version at index 1.
    
    Args:
        tensor: Input numpy array with shape (s1, s2, ..., sn)
        diff: Tuple (d1, d2, ..., dn) specifying the shift amounts for each dimension.
              Cell at (x1, x2, ..., xn) will get content from ((x1-d1) % s1, (x2-d2) % s2, ..., (xn-dn) % sn)
    
    Returns:
        numpy array with shape (s1, s2, ..., sn, 2) where:
        - [..., 0] contains the original tensor
        - [..., 1] contains the shifted tensor
    """
    # Create shifted version using numpy.roll
    # To shift so that cell (x1, x2, ...) gets value from (x1-d1, x2-d2, ...),
    # we need to roll by (d1, d2, ...) along axes (0, 1, ...)
    shifted = tensor.copy()
    axes = tuple(range(len(diff)))
    shift_values = diff
    shifted = np.roll(shifted, shift=shift_values, axis=axes)
    
    result = np.stack([tensor, shifted], axis=-1)
    return result


def create_weight_vector(f: float) -> np.ndarray:
    return np.array([1 - f, f])


def normalize_shift(shift: float) -> tuple[int, float]:
    """
    Normalize a shift value to separate integer and fractional parts.
    
    Splits the shift into an integer part and a fractional part in the range [0, 1).
    
    Args:
        shift: Shift value (can be any float)
    
    Returns:
        Tuple (shift_int, shift_frac) where:
        - shift_int is the integer part of the shift (can be negative)
        - shift_frac is the fractional part in the range [0, 1)
    """
    shift_int = int(np.floor(shift))
    shift_frac = shift - shift_int
    
    # Handle negative fractional parts to ensure shift_frac is in [0, 1)
    if shift_frac < 0:
        shift_frac += 1
        shift_int -= 1
    
    return shift_int, shift_frac

def interpolate_last_dimention(t: np.ndarray, shift_frac: float) -> np.ndarray:
    assert t.shape[-1] == 2
    return np.einsum('...i,i->...', t, create_weight_vector(shift_frac)) # Collapse last dimention into inner product.

def interpolate_shift_dimension_fractional(tensor: np.ndarray, axis: int, shift_frac: float) -> np.ndarray:
    original_dtype = tensor.dtype
    
    # Convert to float to avoid overflow in weighted interpolation
    result = tensor.astype(np.float64)
    
    # Perform linear interpolation along the specified axis with fractional part
    if shift_frac != 0:
        # Create shift tuple: shift by 1 only along the specified axis
        shift_tuple = [0] * tensor.ndim
        shift_tuple[axis] = 1
        result = add_shifted_dimension(result, tuple(shift_tuple))
        result = interpolate_last_dimention(result, shift_frac)
    
    # Convert back to original dtype, handling uint8 overflow properly
    if original_dtype == np.uint8:
        result = np.clip(result, 0, 255).astype(original_dtype)
    else:
        result = result.astype(original_dtype)
    
    return result


def interpolate_shift_dimension(tensor: np.ndarray, axis: int, shift: float) -> np.ndarray:
    """
    Perform linear interpolation along a specific dimension with fractional shift.
    
    For each position along the specified axis, samples from a fractional position
    using linear interpolation between the two nearest integer positions.
    
    Args:
        tensor: Input tensor of any shape
        axis: The dimension/axis along which to interpolate (0-indexed)
        shift: Fractional shift amount. Positive values shift in the positive direction.
               For position i, samples from position (i - shift).
    
    Returns:
        Tensor of the same shape as input with interpolation applied along the specified axis
    """
    # Normalize shift into integer and fractional parts
    shift_int, shift_frac = normalize_shift(shift)
    
    result = tensor.copy()
    
    # Apply integer shift first if needed
    if shift_int != 0:
        result = np.roll(result, shift=shift_int, axis=axis)
    
    # Perform fractional interpolation (shift_frac is now guaranteed to be in [0, 1))
    result = interpolate_shift_dimension_fractional(result, axis, shift_frac)
    
    return result


def bilinear_interpolate_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Perform bilinear interpolation on an image with fractional index shifts.
    
    For each pixel at position (i, j), samples from position (i - dx, j - dy) using
    bilinear interpolation from the 4 nearest integer pixel positions.
    
    Args:
        img: Input image tensor of shape (height, width, channels) or (height, width)
        dx: Fractional shift in the first dimension (rows/x-axis)
        dy: Fractional shift in the second dimension (columns/y-axis)
    
    Returns:
        Image tensor of the same shape as input with bilinear interpolation applied
    """
    result = img.copy()
    
    # First interpolate along the x-axis (axis 0 - rows)
    result = interpolate_shift_dimension(result, axis=1, shift=dx)
    
    # Then interpolate along the y-axis (axis 1 - columns)
    result = interpolate_shift_dimension(result, axis=0, shift=dy)
    
    return result


def interactive_shift(img: np.ndarray):
    """
    Interactive function to shift an image by clicking and dragging the mouse.
    
    Click and drag on the image to shift it. The shift amount is determined by
    the drag distance, allowing for both fractional and large shifts.
    
    Args:
        img: Input image tensor to shift interactively
    """
    fig, ax = plt.subplots()
    im_display = ax.imshow(img)
    ax.axis('off')
    ax.set_title('Click and drag to shift image. Press q to quit.')
    
    # Store original image and current shift state
    original_img = img.copy()
    current_dx = 0.0
    current_dy = 0.0
    drag_start_pos = None
    drag_start_shift = (0.0, 0.0)
    is_dragging = False
    
    def on_press(event):
        nonlocal drag_start_pos, drag_start_shift, is_dragging
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:  # Left mouse button
            drag_start_pos = (event.xdata, event.ydata)
            drag_start_shift = (current_dx, current_dy)
            is_dragging = True
    
    def on_motion(event):
        nonlocal is_dragging
        if not is_dragging or event.inaxes != ax or drag_start_pos is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        # Calculate shift from drag distance
        # Note: event.xdata/ydata are in data coordinates (pixel coordinates)
        # For images, y-axis is inverted, so we need to account for that
        dx = event.xdata - drag_start_pos[0]  # horizontal shift
        dy = event.ydata - drag_start_pos[1]  # horizontal shift
        
        # Calculate total shift: base shift from previous drags + current drag offset
        new_dx = drag_start_shift[0] + dx
        new_dy = drag_start_shift[1] + dy
        
        # Apply shift and update display
        shifted_img = bilinear_interpolate_shift(original_img, new_dx, new_dy)
        im_display.set_array(shifted_img)
        fig.canvas.draw_idle()
    
    def on_release(event):
        nonlocal drag_start_pos, drag_start_shift, is_dragging, current_dx, current_dy
        if not is_dragging or drag_start_pos is None:
            return
        
        if event.inaxes == ax and event.button == 1 and event.xdata is not None and event.ydata is not None:
            # Calculate final shift and update current state
            dx = event.xdata - drag_start_pos[0]
            dy = event.ydata - drag_start_pos[1]
            current_dx = drag_start_shift[0] + dx
            current_dy = drag_start_shift[1] + dy
        
        drag_start_pos = None
        drag_start_shift = (0.0, 0.0)
        is_dragging = False
    
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()


def bottom_half_circle_mask(sx: int, sy: int, mx: int, my: int, r: float) -> np.ndarray:
    """
    Create a binary (sx, sy) mask with ones in the bottom half of a circle.

    Parameters
    ----------
    sx, sy : int
        Image dimensions.
    mx, my : float or int
        Circle center.
    r : float or int
        Circle radius.

    Returns
    -------
    mask : ndarray, shape (sx, sy), dtype=np.uint8
        Binary mask (0 or 1).
    """
    y, x = np.ogrid[:sx, :sy]

    dist_sq = (x - mx)**2 + (y - my)**2
    circle = dist_sq <= r**2
    bottom_half = y >= my

    mask = circle & bottom_half
    return mask.astype(np.uint8)

def get_inverse_rot_matrix(angle: float) -> np.ndarray:
    return np.stack([np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])])

def interpolated_advanced_indexing(t: np.ndarray, source_indices_tensor: np.ndarray) -> np.ndarray:
    sx, sy = t.shape
    assert source_indices_tensor.shape == (2, sx, sy)

    intr_edge_images_ls: list[np.ndarray] = []
    for x_rounding in [np.floor, np.ceil]:
        y_edge_images: list[np.ndarray] = []
        for y_rounding in [np.floor, np.ceil]:
            image_indices = np.stack([
                x_rounding(source_indices_tensor[0, ...]),
                y_rounding(source_indices_tensor[1, ...])
            ], axis=0).astype(np.uint64)
            print(image_indices.shape)
            assert image_indices.shape == (2, sx, sy)

            # Deal with out of bounds values by cycling ends:
            for dim in range(image_indices.shape[0]):
                image_indices[dim,...] %= image_indices.shape[1+dim]

            edge_img = t[image_indices[0], image_indices[1]]
            assert edge_img.shape == (sx, sy)
            
            y_edge_images.append(edge_img)
        
        intr_edge_images_ls.append(np.stack([y_edge_images[0], y_edge_images[1]], axis=-1))

    intr_edge_images = np.stack([intr_edge_images_ls[0], intr_edge_images_ls[1]], axis=-1)
    assert intr_edge_images.shape == (sx, sy, 2, 2)
    
    intr_source_weights = np.stack([1-(source_indices_tensor%1), source_indices_tensor%1], axis=0)
    assert intr_source_weights.shape == (2, 2, sx, sy)

    return np.einsum('...ij,ji...->...', intr_edge_images, intr_source_weights)

def rot(t: np.ndarray, angle: float, intr_factions: bool) -> np.ndarray:
    sx, sy = t.shape # For asserts
    
    inverse_rot_mat = get_inverse_rot_matrix(angle)
    dest_indices_tensor = np.indices(t.shape)
    assert dest_indices_tensor.shape == (2, sx, sy)

    source_indices_tensor = np.einsum('ij,jxy->ixy', inverse_rot_mat, dest_indices_tensor)
    assert source_indices_tensor.shape == (2, sx, sy)

    if intr_factions:
        return interpolated_advanced_indexing(t, source_indices_tensor)
    else:
        source_indices_tensor: np.ndarray = np.rint(source_indices_tensor).astype(np.int64)
        
        # Deal with out of bounds values by cycling ends:
        for dim in range(2):
            source_indices_tensor[dim,...] %= t.shape[dim]
            
        return t[source_indices_tensor[0], source_indices_tensor[1]]

def get_brad_mask():
    img = load_image("instructions/Brad.jpg")
    shape = img.shape
    return bottom_half_circle_mask(shape[0], shape[1], 250, 200, 140)

@cli.command("q2_c")
def q2_c():
    img = load_image("instructions/Cameraman.jpg")
    display_image(bilinear_interpolate_shift(img, 170.3, 130.8))

@cli.command("q2_interactive")
@click.argument("fname", default="Cameraman.jpg")
def q2_interactive(fname: str):
    interactive_shift(load_image(f"instructions/{fname}"))

@cli.command("q2_d")
def q2_d():
    mask = get_brad_mask()
    display_image(mask)
    
@cli.command("q2_e")
def q2_e():
    img = load_image("instructions/Brad.jpg")
    mask = get_brad_mask()
    
    display_image(img*mask)

@cli.command("q2_f")
def q2_f():
    img = load_image("instructions/Brad.jpg")
    mask = get_brad_mask()
    
    img = img*mask

    for angle in [np.pi/frac for frac in [3, 4, 2]]: #60, 45, 90
        display_image(rot(img, angle, False))

@cli.command("q2_g")
def q2_f():
    img = load_image("instructions/Brad.jpg")
    mask = get_brad_mask()
    
    img = img*mask

    for angle in [np.pi/frac for frac in [3, 4, 2]]: #60, 45, 90
        display_images_side_by_side(rot(img, angle, False), rot(img, angle, True))

@cli.command("rotate")
def rotate():
    img = load_image("instructions/Brad.jpg")
    display_images_side_by_side(rot(img, 0.3, False), rot(img, 0.3, True))
    

# =====================================================================
#                          Main
# =====================================================================

if __name__ == "__main__":
    cli()
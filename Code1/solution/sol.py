#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple, List
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass as dc
import click
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation
import math
from PIL import Image
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray
import scipy.ndimage
import os

@click.group()
class cli:
    pass

def get_provided_resource(fname: str) -> str:
    path = f"../{fname}"
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing resource file at '{path}'! Am I running from the correct directory?")
    return path

def get_our_resource(fname: str) -> str:
    path = f"our_resource_files/{fname}"
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing resource file at '{path}'! Am I running from the correct directory?")
    return path

# =====================================================================
#                          Question 1 - Part A
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


# =====================================================================
#                          Question 1 - Part B
# =====================================================================

def loadbob():
    mat = loadmat(get_provided_resource("SpongeBob.mat"))
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
    img = load_image(get_provided_resource("Brad.jpg"))
    shape = img.shape
    return bottom_half_circle_mask(shape[0], shape[1], 250, 200, 140)

@cli.command("q2_c")
def q2_c():
    img = load_image(get_provided_resource("Cameraman.jpg"))
    display_image(bilinear_interpolate_shift(img, 170.3, 130.8))

@cli.command("q2_interactive")
@click.argument("fname", default="Cameraman.jpg")
def q2_interactive(fname: str):
    interactive_shift(load_image(get_provided_resource(fname)))

@cli.command("q2_d")
def q2_d():
    mask = get_brad_mask()
    display_image(mask)
    
@cli.command("q2_e")
def q2_e():
    img = load_image(get_provided_resource("Brad.jpg"))
    mask = get_brad_mask()
    
    display_image(img*mask)

@cli.command("q2_f")
def q2_f():
    img = load_image(get_provided_resource("Brad.jpg"))
    mask = get_brad_mask()
    
    img = img*mask

    for angle in [np.pi/frac for frac in [3, 4, 2]]: #60, 45, 90
        display_image(rot(img, angle, False))

@cli.command("q2_g")
def q2_f():
    img = load_image(get_provided_resource("Brad.jpg"))
    mask = get_brad_mask()
    
    img = img*mask

    for angle in [np.pi/frac for frac in [3, 4, 2]]: #60, 45, 90
        display_images_side_by_side(rot(img, angle, False), rot(img, angle, True))

# =====================================================================
#                          Question 3
# =====================================================================

def image_to_vectors(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 3 and img.shape[2] == 3
    return img.reshape(-1, 3).astype(np.float64)

def vectors_to_image(vectors: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    assert vectors.ndim == 2 and vectors.shape[1] == 3
    return vectors.reshape(shape)


def init_codebook(data: np.ndarray, levels: int, rng: np.random.Generator) -> np.ndarray:
    assert data.ndim == 2 and data.shape[1] == 3
    assert levels <= data.shape[0]
    idx = rng.choice(data.shape[0], size=levels, replace=False)
    return data[idx].copy()


def assign_clusters(data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    assert data.ndim == 2 and codebook.ndim == 2
    dists = np.sum((data[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1)


def update_codebook(
    data: np.ndarray,
    labels: np.ndarray,
    levels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    new_codebook = np.empty((levels, data.shape[1]), dtype=np.float64)

    for k in range(levels):
        mask = labels == k
        if np.any(mask):
            new_codebook[k] = data[mask].mean(axis=0)
        else:
            new_codebook[k] = data[rng.integers(0, data.shape[0])]

    return new_codebook


def compute_distortion(data: np.ndarray, codebook: np.ndarray, labels: np.ndarray) -> float:
    diffs = data - codebook[labels]
    return float(np.mean(np.sum(diffs ** 2, axis=1)))

def max_lloyd(
    image: np.ndarray,
    levels: int,
    max_iter: int,
    meps: float | None = None,
    init_codebook_vectors: np.ndarray | None = None,
    seed: int | None = None,
) -> Tuple[List[np.ndarray], List[float], np.ndarray]:
    assert image.ndim == 3 and image.shape[2] == 3
    assert max_iter >= 0

    rng = np.random.default_rng(seed)
    data = image_to_vectors(image)

    if init_codebook_vectors is not None:
        assert init_codebook_vectors.shape == (levels, data.shape[1])
        codebook = init_codebook_vectors.astype(np.float64).copy()
    else:
        codebook = init_codebook(data, levels, rng)

    images: List[np.ndarray] = []
    distortions: List[float] = []

    prev_distortion = np.inf

    for _ in range(max_iter):
        labels = assign_clusters(data, codebook)
        distortion = compute_distortion(data, codebook, labels)

        distortions.append(distortion)

        quantized = codebook[labels]
        quantized = np.clip(np.rint(quantized), 0, 255).astype(image.dtype)
        images.append(vectors_to_image(quantized, image.shape))

        if meps is not None and prev_distortion < np.inf:
            eps = (prev_distortion - distortion) / prev_distortion
            if eps <= meps:
                break

        prev_distortion = distortion
        codebook = update_codebook(data, labels, levels, rng)

    images.append(image)

    return images, distortions, codebook

def validate_init_codebook(
    codebook: np.ndarray | None,
    levels: int,
) -> None:
    if codebook is None:
        return
    assert codebook.shape == (levels, 3)

def parse_levels_or_codebook(
    value: str,
) -> Tuple[int, np.ndarray | None]:
    try:
        levels = int(value)
        assert levels > 0
        return levels, None
    except ValueError:
        path = Path(value)
        assert path.exists() and path.suffix == ".json"

        with path.open("r") as f:
            raw = json.load(f)

        codebook = np.asarray(raw, dtype=np.float64)
        assert codebook.ndim == 2 and codebook.shape[1] == 3
        assert np.all((0.0 <= codebook) & (codebook <= 1.0))

        codebook = codebook * 255.0

        return codebook.shape[0], codebook


def load_image_q3(path: Path) -> np.ndarray:
    assert path.exists()
    img = plt.imread(path)
    assert img.ndim == 3 and img.shape[2] == 3
    return img


def show_side_by_side(original: np.ndarray, quantized: np.ndarray, title: str = "") -> None:
    assert original.shape == quantized.shape

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(quantized)
    axes[1].set_title("Quantized")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_distortion(ax: plt.Axes, distortions: List[float]) -> None:
    ax.plot(range(len(distortions)), distortions, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distortion")
    ax.set_title("Distortion vs Iteration")
    ax.grid(True)

def show_image_sequence(
    images: List[np.ndarray],
    titles: List[str] | None = None,
    distortions: List[float] | None = None,
) -> None:
    n_img = len(images)
    n = n_img + (1 if distortions is not None else 0)
    assert n > 0

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    idx = 0

    if distortions is not None:
        plot_distortion(axes.flat[idx], distortions)
        idx += 1

    for i in range(n_img):
        ax = axes.flat[idx]
        ax.imshow(images[i])
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i])
        idx += 1

    for j in range(idx, rows * cols):
        axes.flat[j].axis("off")

    plt.tight_layout()
    plt.show()


def quantize_steps(
    image_path: str,
    levels_or_codebook: str,
    max_iter: int = 10,
    meps: float | None = None,
    seed: int = 0,
) -> None:
    path = Path(image_path)
    image = load_image_q3(path)

    levels, init_codebook_vectors = parse_levels_or_codebook(levels_or_codebook)
    validate_init_codebook(init_codebook_vectors, levels)

    images, distortions, _ = max_lloyd(
        image=image,
        levels=levels,
        max_iter=max_iter,
        meps=meps,
        init_codebook_vectors=init_codebook_vectors,
        seed=seed,
    )

    titles = [f"iter {i}" for i in range(len(images) - 1)] + ["original"]

    show_image_sequence(
        images=images,
        titles=titles,
        distortions=distortions,
    )

@cli.command("q3_b")
def q3_b():
    for n_vectors in [6, 15]:
        quantize_steps(get_provided_resource("colorful.tif"), n_vectors, meps=0.02)

@cli.command("q3_d")
def q3_d():
    for init in [get_our_resource("max_lloyed_iv.json"), 9]:
        quantize_steps(get_provided_resource("colorful.tif"), init)

# =====================================================================
#                          Question 4
# =====================================================================

heisenberg = plt.imread(get_provided_resource("heisenberg.jpg"))

#section a:
@cli.command("q4_a")
def q4_a():
    h_entropy = shannon_entropy(heisenberg,base=2)
    print(f"The entropy of Heisenberg is: {h_entropy}")

#section b:
# Below is a github implementing of a huffman encoder for text, with some changes so that it will work for a grayscale image (though we still encode using text).
# source: https://github.com/ybruce61414/Huffman-Code/blob/master/HuffmanCode.ipynb

class Huffman_node():
    def __init__(self,cha,freq):
        self.cha = cha
        self.freq = freq
        self.Lchild = None
        self.Rchild = None
   
    def __repr__(self):
            return '(node object %s:%d)' % (self.cha,self.freq)
        
class HuffmanCoding():
    def __init__(self,text):
        self.root = None
        self.text = text
        self.nodedic = {}
        self.huffcodes = {}
        self.encodes = []
        self.decodes = []
                
    #------ generating huffman tree -------   
    def generate_tree(self):
        self.generate_node() 
        while len(self.nodedic) != 1:
            min_node1 = self.find_minNode()
            min_node2 = self.find_minNode()
            self.root = self.merge_nodes(min_node1,min_node2)
        return self.root              
        
    #---- function set for generating huffman tree -----
    def character_freq(self):
        #generate dic-{cha:freq}
        count = {}
        for cha in self.text:
            count.setdefault(cha,0)
            count[cha] += 1
        return count     

    def generate_node(self):
        #generate dic-{freq:node}
        c = self.character_freq()
        #storing each cha & freq into huffmanNode
        for k,v in c.items():
            newnode = Huffman_node(k,v)
            #multiple value for the same key
            #dic-{key:[ob1,ob2..]}
            self.nodedic.setdefault(v,[]).append(newnode)
        return self.nodedic
    
    def find_minNode(self):
        keys = list(self.nodedic.keys())
        minkey, minlist = keys[0], self.nodedic[keys[0]]
        for k,v in self.nodedic.items():
            if minkey > k:
                minkey,minlist = k,v
        minvalue = minlist.pop(0)
        if not minlist:
            #empty list,delete the minNode from dic
            del self.nodedic[minkey]    
        #return minNode object
        return minvalue 
    
    def merge_nodes(self,min1,min2):
        newnode = Huffman_node(None,min1.freq + min2.freq)
        newnode.Lchild,newnode.Rchild = min1,min2
        #adding newnode into self.nodedic
        self.nodedic.setdefault(min1.freq + min2.freq,[]).append(newnode) 
        return newnode
    
    #----------generating huffman code-----------
    def generate_huffcode(self):
        code = ''
        if self.root != None:
            return self.rec_generate_huffcode(self.root,code)         
            
    def rec_generate_huffcode(self,cur_node,codestr):
        if not cur_node.Lchild and not cur_node.Rchild:
            self.huffcodes[cur_node.cha] = codestr  
        if cur_node.Lchild:
            self.rec_generate_huffcode(cur_node.Lchild,codestr + '0')
        if cur_node.Rchild:
            self.rec_generate_huffcode(cur_node.Rchild,codestr + '1')
         
    #----------------compression-------------------
    def encode(self):
        for cha in self.text:
            self.encodes.append(self.huffcodes[cha])
        #strings in list merge into one string    
        self.encodes = ''.join(self.encodes)
        #turn encodes into string
        return self.encodes     
        
    #----------------decompression------------------
    def decode(self):
        temp_str,temp_dic = '',{}
        #reverse huffcodes
        for k,v in self.huffcodes.items():
            temp_dic[v] = k
        
        for binary_code in self.encodes:
            temp_str += binary_code
            if temp_str in temp_dic.keys():
                self.decodes.append(temp_dic[temp_str])
                temp_str = ''
        self.decodes = ''.join(self.decodes)         
        return self.decodes 

def huffman_encoder(im):
    #the github huffman encoder uses text, so we need to convert the image to a utf format.
    flatten_im = im.flatten().astype(np.int16) #int16 explained below, int8 is insufficient for difference encoding
    input = flatten_im.tobytes().decode("utf-16-le") #decode image into text for huffman encoder, we use np.int16 and utf-16 so that it will handle values from (-255,255), in total 511 bits
    huffman = HuffmanCoding(input)
    huffman.generate_tree()
    huffman.generate_huffcode()
    code = huffman.encode()
    dict = huffman.huffcodes
    char_freq = huffman.character_freq()
    avglen = sum(len(dict[character])*freq for character,freq in char_freq.items())/len(input)
    return code,dict,avglen

# section c
@cli.command("q4_c")
def q4_c():
    heisenberg_code,heisenberg_dict,heisenberg_avglen = huffman_encoder(heisenberg)
    # code length:
    heis_codelength = len(heisenberg_code)
    print(f"Heisenberg Huffman code length is {heis_codelength}")
    # our original image is 8 bits, so compression ratio is 8/avglen
    heis_compression_ratio = 8/heisenberg_avglen
    print(f"Heisenberg Huffman compression ratio is {heis_compression_ratio}")


#section d
def huffman_decoder(code,dict,width,height):
    huffman = HuffmanCoding("")
    huffman.encodes=code
    huffman.huffcodes=dict
    output=huffman.decode()
    output_flat = output.encode("utf-16-le")
    output_numpy=np.frombuffer(output_flat,dtype=np.int16) # again, we decode using int16 since we encoded with it
    return output_numpy.reshape((width,height))


@cli.command("q4_d")
def q4_d():
    heisenberg_code,heisenberg_dict,heisenberg_avglen = huffman_encoder(heisenberg)
    decoded_heisenberg = huffman_decoder(heisenberg_code,heisenberg_dict,heisenberg.shape[0],heisenberg.shape[1])
    heisenberg_mse = np.mean((decoded_heisenberg-heisenberg)**2)
    print(f"MSE between decoded image and original image is: {heisenberg_mse}")

# section e
@cli.command("q4_e")
def q4_e():
    mauritius_rgb= plt.imread(get_provided_resource("mauritius.jpg"))
    mauritius =(rgb2gray(mauritius_rgb)*255).astype(np.uint8) #because rgb2gray produces floats between 0 and 1, we want greyscale between 0 and 255

    mauritius_entropy=shannon_entropy(mauritius,base=2)
    print(f"The entropy of Mauritius is: {mauritius_entropy}")
    mauritius_code,mauritius_dict,mauritius_avglen = huffman_encoder(mauritius)
    print(f"The information rate of Mauritius is: {mauritius_avglen}")

# section f
def zigzag_ver2(img):
    zigzaged = np.concatenate([np.diagonal(img[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-img.shape[0], img.shape[0]+abs(img.shape[0]-img.shape[1])+1)])
    return zigzaged

def un_zigzag(vec_img, M, N):
  output = np.zeros((M,N))
  assert M*N == len(vec_img), f"A vector of size {len(vec_img)} cannot be rearranged into an ({M},{N}) matrix."
  # create indices vector
  indices = np.arange(len(vec_img))
  # rearrange the indices the same way as the original matrix
  indices = indices.reshape((M, N))
  indices = zigzag_ver2(indices)
  # for each element in the vector, replace in the original matrix index
  for (k,element) in zip(indices,vec_img):
    i = int(np.floor(k/N)) # row
    j = int(np.mod(k,N)) # col
    output[i,j]=element # placement
  return output

@cli.command("q4_f")
def q4_f():
    scotland_rgb = plt.imread(get_provided_resource("scotland.jpg"))
    scotland = (rgb2gray(scotland_rgb)*255).astype(np.int16) # because rgb2gray produces floats between 0 and 1, we want greyscale between 0 and 255
    scot_column_stack = scotland.flatten(order='F')

    scot_zigzag = zigzag_ver2(scotland)
    column_diff = np.zeros_like(scot_column_stack,dtype=np.int16)
    column_diff[0] = scot_column_stack[0]
    zigzag_diff = np.zeros_like(scot_zigzag,dtype=np.int16)
    zigzag_diff[0] = scot_zigzag[0]
    column_diff[1:]=[scot_column_stack[i]-scot_column_stack[i-1] for i in range(1,len(scot_column_stack))]
    zigzag_diff[1:]=[scot_zigzag[i]-scot_zigzag[i-1] for i in range(1,len(scot_zigzag))]
    column_image = column_diff.reshape(scotland.shape,order='F')
    zigzag_image = un_zigzag(zigzag_diff,scotland.shape[0],scotland.shape[1])
    column_hist = np.histogram(column_image, bins=511,range=(-255,256))[0]
    zigzag_hist = np.histogram(zigzag_image, bins=511,range=(-255,256))[0]

    #plotting the histograms
    plt.figure()
    plt.bar(np.arange(-255, 256),column_hist,width=1)
    plt.title("Column Difference")
    plt.xlabel("Difference")
    plt.ylabel("Amount")
    plt.show()

    plt.figure()
    plt.bar(np.arange(-255,256),zigzag_hist,width=1)
    plt.title("Zigzag Difference")
    plt.xlabel("Difference")
    plt.ylabel("Amount")
    plt.show()

    column_code, column_dict, column_avglen = huffman_encoder(column_image)
    zigzag_code, zigzag_dict, zigzag_avglen = huffman_encoder(zigzag_image)
    print(f"avglen of column stacked image is {column_avglen}")
    print(f"avglen of zigzag image is {zigzag_avglen}")

# =====================================================================
#                          Question 5
# =====================================================================

# section a:
large_num=99999999
def calc_SSD(image,template):
    img_double = np.asarray(image,dtype=np.float64) # the question tells us to use double which is float64.
    temp_double = np.asarray(template,dtype=np.float64)
    M=temp_double.shape[0]
    N=temp_double.shape[1]
    # according to formula we can expand the square as sumI^2-2*sumI*T+sumT^2
    # where sumT^2 is a constant (every entry is just T^2(i,j)), and 2*sum*I*T is 2D spatial correlation of T with I,
    # and sumI^2 is a 2D spatial correlation of I^2(x+i-M/2-1,y+j-N/2-1) with a kernel of all ones because the number one is neutral to multiplication.
    sumT2 = np.sum(temp_double**2)
    sumI2 = scipy.ndimage.correlate(img_double**2,np.ones(temp_double.shape))
    sumIT = scipy.ndimage.correlate(img_double,temp_double)

    S=sumT2+sumI2-2*sumIT
    # now we set the areas where the template doesn't fit to a large number
    S[-M//2:,:]=large_num
    S[:M//2,:]=large_num
    S[:,-N//2:]=large_num
    S[:,:N//2]=large_num
    return S

# section b:
poe_img = plt.imread(get_provided_resource("Text.jpg"))

@cli.command("q5_b")
def q5_b():
    e10 = plt.imread(get_provided_resource("E10.jpg"))
    e11 = plt.imread(get_provided_resource("E11.jpg"))
    e12 = plt.imread(get_provided_resource("E12.jpg"))
    e14 = plt.imread(get_provided_resource("E14.jpg"))
    e16 = plt.imread(get_provided_resource("E16.jpg"))
    imgs = [e10,e11,e12,e14,e16]
    titles = ["E10","E11","E12","E14","E16"]
    fig,axes = plt.subplots(1,5,figsize=(20,6))
    for i,axis in enumerate(axes):
        axis.imshow(imgs[i],cmap='gray')
        axis.set_title(titles[i])
        axis.axis('off')
    plt.tight_layout()
    plt.show()

    # algorithm to find best match:
    # we just find the template with the lowest SSD.
    # however, we need to be careful, we must normalize the sizes of the templates because we need to sum the same amount of numbers for each template to compare reliably.
    ssds=[]
    for img in imgs:
        ssds.append(calc_SSD(poe_img,img))
    sizes=[]
    for img in imgs:
        sizes.append(img.size)
    def best_match(list_ssds,list_sizes):
        minimum_list =[np.min(ssd) for ssd in list_ssds]
        result_list = [minimum_list[i]/list_sizes[i] for i in range(len(list_ssds))]
        return np.argmin(result_list)


    best=best_match(ssds,sizes)
    print(f"best match is {titles[best]}")


# section c
@cli.command("q5_c")
def q5_c():
    # Now, we load the 4 templates we cropped from the image and calculate SSDs
    temp_a = plt.imread(get_our_resource("tempa.jpg"))
    temp_A = plt.imread(get_our_resource("tempacap.jpg"))
    temp_t = plt.imread(get_our_resource("tempt.jpg"))
    temp_T = plt.imread(get_our_resource("temptcap.jpg"))
    # we cropped the images with MSPaint and we saved as JPG so we need to convert to gray levels
    temps = [temp_a,temp_A,temp_t,temp_T]
    for i in range(len(temps)):
        temps[i] = np.dot(temps[i][...,:3],[0.299, 0.587, 0.114]) #this converts to grayscale

    a_ssd = calc_SSD(poe_img,temps[0])
    A_ssd = calc_SSD(poe_img,temps[1])
    t_ssd = calc_SSD(poe_img,temps[2])
    T_ssd = calc_SSD(poe_img,temps[3])

    # let's normalize SSDs, so that we can get a unified treshold

    a_ssd_nor = a_ssd/temp_a.size
    A_ssd_nor = A_ssd/temp_A.size
    t_ssd_nor = t_ssd/temp_t.size
    T_ssd_nor = T_ssd/temp_T.size

    # here are our templates:

    titles = ["template a","template A","template t","template T"]
    fig,axes = plt.subplots(1,4,figsize=(20,6))
    for i,axis in enumerate(axes):
        axis.imshow(temps[i],cmap='gray')
        axis.set_title(titles[i])
        axis.axis('off')
    plt.tight_layout()
    plt.show()

    # now we need to determine the threshold. for that, we plot the flattened and sorted SSD images without the large borders:

    ssd_imgs = [np.sort(a_ssd_nor[a_ssd_nor<large_num/temp_a.size]),np.sort(A_ssd_nor[A_ssd_nor<large_num/temp_A.size]),np.sort(t_ssd_nor[t_ssd_nor<large_num/temp_t.size]),np.sort(T_ssd_nor[T_ssd_nor<large_num/temp_T.size])]
    titles = ["a ssd","A ssd","t ssd","T ssd"]
    fig,axes = plt.subplots(1,4,figsize=(30,7))
    for i,axis in enumerate(axes):
        #let's zoom in only on the best matches. since the text is not long, I estimate every letter appears at most ~50 times.
        axis.plot(ssd_imgs[i][:50])
        axis.set_title(titles[i])
        axis.axis('on')
        axis.grid(True)
    plt.tight_layout()
    plt.show()

    # from this, we see that the number of appearances is at the point where the slope becomes very high. this is because the error increases very much
    # once it cannot find any more matches.
    # so, once we find a spot where the derivative suddenly becomes very large, then the erorr increased because it didn't find more
    # matches, then we know we found the number of occurences

    derivative_a = np.diff(ssd_imgs[0])
    derivative_A = np.diff(ssd_imgs[1])
    derivative_t = np.diff(ssd_imgs[2])
    derivative_T = np.diff(ssd_imgs[3])

    # Let's plot the derivatives to determine threshold

    derivatives = [derivative_a,derivative_A,derivative_t,derivative_T]
    titles = ["a derivative","A derivative","t derivative","T derivative"]
    fig,axes = plt.subplots(1,4,figsize=(30,7))
    for i,axis in enumerate(axes):
        axis.plot(derivatives[i][:50])
        axis.set_title(titles[i])
        axis.axis('on')
        axis.grid(True)
    plt.tight_layout()
    plt.show()

    threshold=270
    #we add 1 to the result, because the derivative becomes very large at index i, which means the amount of numbers is i+1
    occur_a = np.where(derivative_a>threshold)[0][0]+1
    occur_A = np.where(derivative_A>threshold)[0][0]+1
    occur_t = np.where(derivative_t>threshold)[0][0]+1
    occur_T = np.where(derivative_T>threshold)[0][0]+1
    print(f"Occurences of a: {occur_a}")
    print(f"Occurences of A: {occur_A}")
    print(f"Occurences of t: {occur_t}")
    print(f"Occurences of T: {occur_T}")

    # for the next part we load template of 'c' and template of 'k'
    temp_c = plt.imread(get_provided_resource("c.jpg"))
    temp_k = plt.imread(get_provided_resource("k.jpg"))
    # let's calculate SSD of temp_c with the edgar allan poe image

    ssd_c = calc_SSD(poe_img,temp_c)
    c_ssd_nor = ssd_c/temp_c.size
    ssd_c_adj = np.sort(c_ssd_nor[c_ssd_nor<large_num/temp_c.size])
    #let's see derivative graph to find a threshold
    derivative_c = np.diff(ssd_c_adj)


    # Let's plot the derivatives to determine threshold

    derivatives2 = [derivative_c]
    titles2 = ["c derivative"]
    fig,axis=plt.subplots(1,1,figsize=(30, 7))
    axis.plot(derivatives2[0][:50])
    axis.set_title(titles2[0])
    axis.axis('on')
    axis.grid(True)
    plt.tight_layout()
    plt.show()

    #500 is a good threshold from this graph.
    limit2=500
    #to avoid loops as much as possible we will create a mask that finds pixels whose error is below limit2, to find all places where a 'c' is.
    mask=(c_ssd_nor<limit2)
    c_y,c_x = np.where(mask)
    poe_img_new=poe_img.copy()
    for i in range(len(c_y)): # place the template k instead of c for all coordinates
        y1=c_y[i]-temp_k.shape[0]//2
        y2=y1+temp_k.shape[0]
        x1=c_x[i]-temp_k.shape[1]//2
        x2=x1+temp_k.shape[1]
        poe_img_new[y1:y2,x1:x2]=temp_k
    # display:
    plt.figure(figsize=(20,7))
    plt.imshow(poe_img_new,cmap='gray')
    plt.title("Replaced c with k")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# =====================================================================
#                          Main
# =====================================================================

if __name__ == "__main__":
    cli()
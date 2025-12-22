#!/usr/bin/python3
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path: str) -> np.ndarray:
    """
    Load a JPG image file and return it as a numpy array.
    
    Args:
        path: Path to the JPG image file
        
    Returns:
        numpy array with shape (height, width, 3) for RGB images,
        or (height, width) for grayscale images. Values are in range [0, 255]
        with dtype uint8.
    """
    img = Image.open(path)
    # Convert to RGB if necessary (handles RGBA, P, L modes)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def display_image(img: np.ndarray):
    """
    Display an image tensor as a matplotlib figure.
    
    Args:
        img: numpy array with shape (height, width, 3) for RGB images,
             or (height, width) for grayscale images. Values should be in 
             range [0, 255] with dtype uint8.
    """
    plt.imshow(img)
    plt.axis('off')  # Hide axes for cleaner display
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
    
    # Stack original and shifted along new axis
    result = np.stack([tensor, shifted], axis=-1)
    return result


def collapse_last_dimension(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Collapse the last dimension of a tensor by computing inner products with a vector.
    
    Args:
        t: Input tensor of shape (dims..., n) where n is the size of the last dimension
        v: Vector of length n
    
    Returns:
        Tensor of shape (dims...) where each cell (x1, ..., xk) contains the inner
        product of t[x1, ..., xk, :] with v.
    """
    return np.einsum('...i,i->...', t, v)


def create_weight_vector(f: float) -> np.ndarray:
    """
    Create a 2-element numpy array from a float value.
    
    Args:
        f: Float value
    
    Returns:
        numpy array of shape (2,) with values [1-f, f]
    """
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
    # Extract integer and fractional parts
    shift_int = int(np.floor(shift))
    shift_frac = shift - shift_int
    
    # Handle negative fractional parts to ensure shift_frac is in [0, 1)
    if shift_frac < 0:
        shift_frac += 1
        shift_int -= 1
    
    return shift_int, shift_frac


def interpolate_shift_dimension_fractional(tensor: np.ndarray, axis: int, shift_frac: float) -> np.ndarray:
    """
    Perform linear interpolation along a specific dimension with fractional shift in range [0, 1).
    
    Assumes shift_frac is in the range [0, 1). For each position along the specified axis,
    samples from a fractional position using linear interpolation between the two nearest
    integer positions.
    
    Args:
        tensor: Input tensor of any shape
        axis: The dimension/axis along which to interpolate (0-indexed)
        shift_frac: Fractional shift amount in range [0, 1). For position i, samples from
                   position (i - shift_frac) using interpolation between i and i-1.
    
    Returns:
        Tensor of the same shape as input with interpolation applied along the specified axis
    """
    # Store original dtype to convert back later
    original_dtype = tensor.dtype
    
    # Convert to float to avoid overflow in weighted interpolation
    result = tensor.astype(np.float64)
    
    # Perform linear interpolation along the specified axis with fractional part
    if shift_frac != 0:
        # Create shift tuple: shift by 1 only along the specified axis
        shift_tuple = [0] * tensor.ndim
        shift_tuple[axis] = 1
        result = add_shifted_dimension(result, tuple(shift_tuple))
        result = collapse_last_dimension(result, create_weight_vector(shift_frac))
    
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
    result = interpolate_shift_dimension(result, axis=0, shift=dx)
    
    # Then interpolate along the y-axis (axis 1 - columns)
    result = interpolate_shift_dimension(result, axis=1, shift=dy)
    
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
        dy = event.ydata - drag_start_pos[0]  # horizontal shift
        
        # Calculate total shift: base shift from previous drags + current drag offset
        new_dx = drag_start_shift[0] + dx
        new_dy = drag_start_shift[1] + dy
        
        # Apply shift and update display
        shifted_img = bilinear_interpolate_shift(original_img, new_dy, new_dx)
        im_display.set_array(shifted_img)
        fig.canvas.draw_idle()
    
    def on_release(event):
        nonlocal drag_start_pos, drag_start_shift, is_dragging, current_dx, current_dy
        if not is_dragging or drag_start_pos is None:
            return
        
        if event.inaxes == ax and event.button == 1 and event.xdata is not None and event.ydata is not None:
            # Calculate final shift and update current state
            dx = event.xdata - drag_start_pos[0]
            dy = event.ydata - drag_start_pos[0]
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


m = load_image("instructions/scotland.jpg")
print(m.shape)
interactive_shift(m)
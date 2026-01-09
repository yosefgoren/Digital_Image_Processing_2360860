#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from typing import Tuple, List
import click
import matplotlib.pyplot as plt
from pathlib import Path
import math


Array = np.ndarray


def image_to_vectors(img: Array) -> Array:
    assert img.ndim == 3 and img.shape[2] == 3
    return img.reshape(-1, 3).astype(np.float64)


def vectors_to_image(vectors: Array, shape: Tuple[int, int, int]) -> Array:
    assert vectors.ndim == 2 and vectors.shape[1] == 3
    return vectors.reshape(shape)


def init_codebook(data: Array, levels: int, rng: np.random.Generator) -> Array:
    assert data.ndim == 2 and data.shape[1] == 3
    assert levels <= data.shape[0]
    idx = rng.choice(data.shape[0], size=levels, replace=False)
    return data[idx].copy()


def assign_clusters(data: Array, codebook: Array) -> Array:
    assert data.ndim == 2 and codebook.ndim == 2
    dists = np.sum((data[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1)


def update_codebook(
    data: Array,
    labels: Array,
    levels: int,
    rng: np.random.Generator,
) -> Array:
    new_codebook = np.empty((levels, data.shape[1]), dtype=np.float64)

    for k in range(levels):
        mask = labels == k
        if np.any(mask):
            new_codebook[k] = data[mask].mean(axis=0)
        else:
            new_codebook[k] = data[rng.integers(0, data.shape[0])]

    return new_codebook


def compute_distortion(data: Array, codebook: Array, labels: Array) -> float:
    diffs = data - codebook[labels]
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


def max_lloyd(
    image: Array,
    levels: int,
    meps: float,
    max_iter: int = 100,
    seed: int | None = None,
) -> Tuple[Array, List[float], Array]:
    assert image.ndim == 3 and image.shape[2] == 3

    rng = np.random.default_rng(seed)
    data = image_to_vectors(image)

    codebook = init_codebook(data, levels, rng)
    distortions: List[float] = []

    prev_distortion = np.inf

    for _ in range(max_iter):
        labels = assign_clusters(data, codebook)
        distortion = compute_distortion(data, codebook, labels)
        distortions.append(distortion)

        if prev_distortion < np.inf:
            eps = (prev_distortion - distortion) / prev_distortion
            if eps <= meps:
                break

        prev_distortion = distortion
        codebook = update_codebook(data, labels, levels, rng)

    quantized = codebook[labels]
    quantized = np.clip(np.rint(quantized), 0, 255).astype(image.dtype)

    output_image = vectors_to_image(quantized, image.shape)

    return output_image, distortions, codebook

def max_lloyd_progressive(
    image: Array,
    levels: int,
    max_iter: int,
    meps: float | None = None,
    init_codebook_vectors: Array | None = None,
    seed: int | None = None,
) -> Tuple[List[Array], List[float], Array]:
    assert image.ndim == 3 and image.shape[2] == 3
    assert max_iter >= 0

    rng = np.random.default_rng(seed)
    data = image_to_vectors(image)

    if init_codebook_vectors is not None:
        assert init_codebook_vectors.shape == (levels, data.shape[1])
        codebook = init_codebook_vectors.astype(np.float64).copy()
    else:
        codebook = init_codebook(data, levels, rng)

    images: List[Array] = []
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


# * * * Usage * * * 

def load_image(path: Path) -> Array:
    assert path.exists()
    img = plt.imread(path)
    assert img.ndim == 3 and img.shape[2] == 3
    return img


def show_side_by_side(original: Array, quantized: Array, title: str = "") -> None:
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
    images: List[Array],
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


@click.group()
def cli() -> None:
    pass

@cli.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--levels", default=8, show_default=True, type=int)
@click.option("--meps", default=0.02, show_default=True, type=float)
@click.option("--max-iter", default=50, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
def quantize(
    image_path: str,
    levels: int,
    meps: float,
    max_iter: int,
    seed: int,
) -> None:
    path = Path(image_path)
    image = load_image(path)

    dataout, distortion, _ = max_lloyd(
        image=image,
        levels=levels,
        meps=meps,
        max_iter=max_iter,
        seed=seed,
    )

    title = f"levels={levels}, iterations={len(distortion)}"
    show_side_by_side(image, dataout, title=title)
@cli.command(name="quantize-steps")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--levels", default=8, show_default=True, type=int)
@click.option("--max-iter", default=10, show_default=True, type=int)
@click.option("--meps", default=None, type=float)
@click.option("--seed", default=0, show_default=True, type=int)
def quantize_steps(
    image_path: str,
    levels: int,
    max_iter: int,
    meps: float | None,
    seed: int,
) -> None:
    path = Path(image_path)
    image = load_image(path)

    images, distortions, _ = max_lloyd_progressive(
        image=image,
        levels=levels,
        max_iter=max_iter,
        meps=meps,
        seed=seed,
    )

    titles = [f"iter {i}" for i in range(len(images) - 1)] + ["original"]

    show_image_sequence(
        images=images,
        titles=titles,
        distortions=distortions,
    )


if __name__ == "__main__":
    cli()

# Project Proposal: Brightness-Preserving Diffusion for Image Colorization

## Motivation
Image colorization aims to reconstruct plausible chrominance information for grayscale images while preserving their structural and luminance content. Traditional approaches—ranging from CNN-based regression to GANs—often produce desaturated colors or semantic inconsistencies. Recently, diffusion models have emerged as powerful generative priors for image synthesis and conditional generation. However, most diffusion-based colorization methods apply standard Gaussian noise either to the full image or to chrominance channels without explicitly enforcing perceptual constraints such as brightness preservation.

## Main Idea
This project proposes to study **brightness-preserving diffusion processes for image colorization**, where the forward diffusion corrupts **only the chrominance components** of an image while keeping luminance fixed. Instead of isotropic Gaussian noise in color space, the corruption process will be constrained to perceptually meaningful subspaces (e.g., hue rotation or chroma-plane perturbations), thereby respecting constant-brightness manifolds. The reverse diffusion model is trained to iteratively recover plausible color information conditioned on the grayscale image.

## Methodology
Images will be represented in perceptually motivated color spaces such as Lab or HSV. During training, the luminance (L or V) channel will remain unchanged, while controlled noise is applied to chrominance channels following a diffusion schedule. A conditional diffusion model will be trained to predict the denoising direction given the corrupted chroma, the fixed luminance, and the diffusion timestep. At inference time, the model will start from a grayscale image with randomized chrominance initialization and iteratively synthesize color.

## Relation to Prior Work
Recent diffusion-based colorization methods use standard diffusion formulations or latent diffusion models conditioned on grayscale images and, in some cases, additional semantic cues. In contrast, this project focuses explicitly on **modifying the forward diffusion process** to enforce brightness preservation and perceptual constraints, enabling a controlled study of how the choice of corruption affects colorization quality and stability.

## Experimental Plan
The project will reproduce a baseline diffusion-based colorization method and compare it against the proposed brightness-preserving diffusion. Experiments will evaluate visual quality and quantitative metrics (e.g., PSNR, SSIM, perceptual similarity) on standard datasets. Additional ablation studies will examine different color spaces and corruption strategies.

## Expected Contribution
The expected outcome is a clearer understanding of how structured, perceptually constrained diffusion processes impact automatic image colorization. This work aims to provide both empirical results and qualitative insights, contributing to the broader study of diffusion models for low-level and mid-level image processing tasks.


# Project Proposal  
**Course:** Digital Image Processing (2360860)  
**Student:** <Your Name>  
**ID:** <Your ID>  
**Email:** <Your Email>  
**Semester:** Winter 25–26  

---

## Project Title  
**Brightness-Preserving Diffusion Processes for Automatic Image Colorization**

---

## Starting Point and Motivation

This project will start from the recent paper  
**“Multimodal Semantic-Aware Automatic Colorization with Diffusion Prior”**  
(Han Wang et al., 2024).

The paper adapts **Stable Diffusion** for automatic image colorization by conditioning the diffusion process on grayscale input and high-level semantic cues. A key contribution of the paper is *luminance alignment*, implemented by injecting grayscale information into the latent diffusion process and by using a luminance-aware decoder during reconstruction.

While the method achieves strong results, an important observation is that **luminance is guided rather than preserved by construction**. The diffusion process itself still operates on full latent representations using standard Gaussian noise, meaning that brightness information can be temporarily distorted and must later be corrected by conditioning and decoder design.

This project explores whether a more principled alternative is possible.

---

## Main Idea

Instead of guiding luminance through conditioning, this project proposes to **remove luminance from the stochastic diffusion process altogether**.

The central idea is to redesign the forward diffusion process such that:
- The **luminance (brightness) channel remains fixed** throughout training and inference.
- Noise is applied **only to chrominance components** in a perceptually meaningful color space (e.g., Lab or HSV).
- The diffusion model learns to iteratively recover color information while brightness is preserved by construction.

In contrast to the reference paper, where luminance consistency is enforced indirectly, this approach aims to guarantee luminance preservation at every diffusion step.

---

## Relation to the Reference Paper

The proposed work is directly grounded in the pipeline of Wang et al.:

- Both approaches use diffusion models for grayscale-to-color generation.
- Both condition the model on grayscale structure.
- Both aim to reduce artifacts and semantic inconsistencies.

However, the key difference is philosophical and technical:

- **Reference paper:**  
  Luminance is *guided* using latent concatenation and a luminance-aware decoder, but standard Gaussian diffusion is still applied.
- **Proposed project:**  
  Luminance is *preserved by construction* by redefining the diffusion corruption process to act only on chrominance.

This allows a controlled study of whether explicit luminance invariance simplifies learning, improves stability, or reduces artifacts.

---

## Methodology Overview

1. Convert training images to a perceptual color space (e.g., Lab).
2. Keep the luminance channel fixed.
3. Apply a diffusion process only to chrominance channels using different noise designs.
4. Train a conditional diffusion model to denoise chrominance given luminance.
5. Compare results against a baseline diffusion colorization method inspired by the reference paper.

---

## Experimental Plan

- Reproduce or approximate a diffusion-based colorization baseline.
- Implement the proposed brightness-preserving diffusion variant.
- Evaluate results visually and quantitatively (FID, PSNR, colorfulness).
- Conduct ablation studies on color spaces and noise formulations.

---

## Expected Outcome

The project aims to clarify the role of the diffusion corruption process in colorization tasks and to evaluate whether enforcing brightness preservation at the process level offers advantages over luminance guidance alone. The work is expected to produce both qualitative insights and empirical results relevant to diffusion-based image processing.

---

*This proposal will be refined after completing a deeper technical review of the reference paper and obtaining approval for the experimental scope.*

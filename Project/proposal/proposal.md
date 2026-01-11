# Project Proposal  
**Course:** Digital Image Processing (2360860)  
**Student:** <Your Name>  
**ID:** <Your ID>  
**Email:** <Your Email>  
**Semester:** Winter 25–26  

---

## Project Title  
**Brightness-Preserving Diffusion for Automatic Image Colorization**

---

## Motivation

Automatic image colorization is the task of assigning plausible colors to a grayscale image. In recent years, diffusion models have shown strong performance in image generation and have started to be used for colorization as well, mainly by conditioning a diffusion model on a grayscale input.

A recent paper, **“Multimodal Semantic-Aware Automatic Colorization with Diffusion Prior” (Wang et al., 2024)**, adapts Stable Diffusion for this task. The paper introduces luminance conditioning in both the latent diffusion process and the final image reconstruction, and achieves strong visual results.

However, an important observation from this work is that **luminance is guided, not preserved by construction**. The diffusion process itself still uses standard Gaussian noise on latent representations that mix brightness and color, and luminance consistency is enforced indirectly through conditioning and decoder design.

This raises a natural question: instead of guiding luminance throughout the diffusion process, what happens if luminance is removed from the stochastic process altogether?

---

## Main Idea

The main idea of this project is to modify the diffusion process used for image colorization so that **brightness is preserved by construction**.

Rather than applying diffusion noise to full image representations, the proposed approach applies diffusion **only to chrominance channels**, while keeping the luminance channel fixed throughout training and inference. The model is then trained to recover chrominance information conditioned on the unchanged luminance.

This directly contrasts with the reference paper’s approach, where luminance information is repeatedly injected to correct distortions introduced by the diffusion process. Here, the goal is to avoid those distortions in the first place by redefining the forward diffusion process.

---

## Method Overview

- Images are converted to a perceptual color space (e.g., Lab or HSV).
- The luminance channel is kept fixed.
- Noise is applied only to chrominance channels following a diffusion schedule.
- A conditional diffusion model learns to denoise chrominance given luminance.
- At inference time, the model starts from a grayscale image with random chrominance and iteratively synthesizes color.

The architecture and training setup will follow existing diffusion-based colorization methods as closely as possible, with the main change being the definition of the forward diffusion process.

---

## Experimental Plan

- Implement or reproduce a baseline diffusion-based colorization method inspired by the reference paper.
- Implement the proposed brightness-preserving diffusion variant.
- Compare visual quality and quantitative metrics such as FID, PSNR, and colorfulness.
- Perform ablation studies on color space choice and chrominance noise design.

---

## Expected Outcome

The project aims to evaluate whether preserving luminance by construction simplifies the colorization task, reduces artifacts, or improves stability compared to luminance-guided diffusion. The results should provide practical insight into how diffusion process design affects image colorization performance.

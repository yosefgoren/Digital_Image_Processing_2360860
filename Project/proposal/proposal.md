# Digital Image Processing 2360860 w25-26 - Project Proposal
- **Student Name:** Yosef Goren
- **ID:** 211515606
- **Email:** yosefgoren@campus.technion.ac.il

---

# Image Colorization: Luminance-Preserving Diffusion

## Motivation
Image colorization is a classic image processing problem. Unsuprizingly - various ML models have been sucessfully applied to it: CNN Regression Models, GAN's and VAE's.
More recenctly, this list came to include stable-diffusion which have exhibited SOTA results for image generation.

Therefore we belive a key question in this doamin is:

**Q1: How/Should we pivot stable-diffusion architacture to fit image colorization?**

## Research Grounding Point
As a reference point, we consider the paper:

**P1: Multimodal Semantic-Aware Automatic Colorization with Diffusion Prior (Wang et al., 2024)**.

This paper provides an image colorization solution (inference:) which intakes greyscale images and textual directions.
For training - they use a pre-trained image generation diffusion model, and fine tune it with a tweaked architacture:
The tweak introduces luminance conditioning in the reconstruction process by essentially concatenating the greyscale image to the intermidiate result before each pass.

However, an important observation from this work is that **luminance is guided, not preserved by construction**. The diffusion process itself still uses standard Gaussian noise on latent representations that mix luminance and color, and luminance consistency is enforced indirectly through conditioning and decoder design.

This raises a natural question:

**Q2: What happens if luminance is removed from the reconstruction process altogether?**


## Project Suggestion
We suggest performing a quantitative comparison between diffusion models that utilize different techniques for using the greyscale image as a prior - aiming to answer Q1.

Additionally, the specific approach were diffusion is applied **only to chrominance channels**, while keeping the luminance channel fixed will be examined - aiming to answer Q2.


## Method Overview
While it's too early to decide on specific methodology, we imagine answering Q2 might look like:
- Pick one of the following:
    * Taking a pre-trained image generation stable-diffusion model and adapting it to a form where only luminance is provided to it.
    * Training a model from scratch.
- Traning/Fine-tuning it, iteratively:
    * Take a colored image.
    * Apply noise in the chrominance space to the image (luminance fixed).
    * Require the model to restore colors.

Wether we use a base-model or not - it will be structurally incapable of modifying the luminance.

For inference:
- Take a greyscale image.
- Apply noise in the chrominance space to it.
- Apply the model to it iteratively.

## Practical Notes
We have struggled to find a reference paper with an implementation that is actually accessible.
This might mean our work will be based on a different work to ensure this project maintains a resonable scale and does not turn into a coding project.
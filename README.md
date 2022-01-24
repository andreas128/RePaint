# RePaint
**Inpainting using Denoising Diffusion Probabilistic Models** [[Paper]](https://bit.ly/3rPeXb2)

![Denoising_Diffusion_Inpainting_Animation](https://user-images.githubusercontent.com/11280511/150849757-5cd762cb-07a3-46aa-a906-0fe4606eba3b.gif)



<br>

# Code will be released soon

Star this project to get notified.

<p align="center">
  <a href="#"><img src="https://user-images.githubusercontent.com/11280511/150838003-01136487-6671-4cc4-ae4d-bef8e60c605e.png"></a>
</p>


<br>

# How RePaint fills a missing image part using diffusion models

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td><img alt="RePaint Inpainting using Denoising Diffusion Probabilistic Models Demo 1" src="https://user-images.githubusercontent.com/11280511/150766080-9f3d7bc9-99f2-472e-9e5d-b6ed456340d1.gif"></td>
        <td><img alt="RePaint Inpainting using Denoising Diffusion Probabilistic Models Demo 2" src="https://user-images.githubusercontent.com/11280511/150766125-adf5a3cb-17f2-432c-a8f6-ce0b97122819.gif"></td>
  </tr>
</table>

**What are the blue parts?** <br>
Those parts are missing and therefore have to be filled by RePaint. <br> RePaint generates the missing parts inspired by the known parts.

**How does it work?** <br>
RePaint starts from pure noise. Then the image is denoised step-by-step.  <br> It uses the known part to fill the unknown part in each step.

**Why does the noise level fluctuate during generation?** <br>
Our noise schedule improves the harmony between the generated and <br> the known part [[4.2 Resampling]](https://bit.ly/3fTPs2T).


<br>

# RePaint on diverse content and shapes of missing regions

The blue region is unknown and filled by RePaint:

![Denoising Diffusion Probabilistic Models Inpainting](https://user-images.githubusercontent.com/11280511/150803812-a4729ef8-6ad4-46aa-ae99-8c27fbb2ea2e.png)


**Note: RePaint creates many meaningful fillings.** <br>
1) **Face:** Expressions and features like an earring or a mole. <br>
2) **Computer:** The computer screen shows different images, text, and even a logo. <br>
3) **Greens:** RePaint makes sense of the tiny known part and incorporates it in a beetle, spaghetti, and plants. <br>
4) **Garden:** From simple filling like a curtain to complex filling like a human. <br>


<br>

# Extreme Case 1: Generate every second line
![Denoising_Diffusion_Probabilistic_Models_Inpainting_Every_Second_Line](https://user-images.githubusercontent.com/11280511/150818064-29789cbe-73c7-45de-a955-9fad5fb24c0e.png)

- Every Second line of the input image is unknown.
- Most inpainting methods fail on such masks.


<br>

# Extreme Case 2: Upscale an image
![Denoising_Diffusion_Probabilistic_Models_Inpainting_Super_Resolution](https://user-images.githubusercontent.com/11280511/150818741-5ed19a0b-1cf8-4f28-9e57-2e4c12303c3e.png)

- The inpainting only knows pixels with a strided access of 2.
- A ratio of 3/4 of the image has to be filled.
- This is equivalent to Super-Resolution with Nearest Neighbor kernel.

<br>

# How RePaint conditions the diffusion model on the known part

- RePaint uses Denoising Diffusion Probabilistic Models.
- We condition this process on the given image content.

![Denoising_Diffusion_Probabilistic_Models_Inpainting_Method](https://user-images.githubusercontent.com/11280511/150822344-8e71070b-936d-4c5b-9298-ac29af266990.png)

**Intuition of one denoising step:**
1) **Sample the known part:** Add gaussian noise to the known regions of the image. <br> We obtain a noisy image that follows the denoising process exactly.
2) **Denoise one step:** Denoise the previous image for one step. This generates  <br> content for the unknown region conditioned on the known region.
3) **Join:** Merge the images from both steps.

Details are in Algorithm 1 on Page 4. [[Paper]](https://bit.ly/3IwCPH7)


<br>

# How to harmonize the generated with the known part?

- **Fail:** When using only the algorithm above, the filling is not well harmonized with the known part (n=1).
- **Fix:** When applying the [[4.2 Resampling]](https://bit.ly/358b9tN) technique, the images are better harmonized (n>1).

<img width="1577" alt="Diffusion Model Resampling" src="https://user-images.githubusercontent.com/11280511/150822917-737c00b0-b6bb-439d-a5bf-e73238d30990.png">

<br>

# RePaint Fails
- The ImageNet model is biased towards inpainting dogs.
- This is due to the high ratio of dog images in ImageNet.

<img width="1653" alt="RePaint Fails" src="https://user-images.githubusercontent.com/11280511/150853163-b965f59c-5ad4-485b-816e-4391e77b5199.png">

<br>

# User Study State-of-the-Art Comparison

- Outperforms autoregressive-based and GAN-based SOTA methods for <br> all masks with significance 95% except for two inconclusive cases.
- The user study was done for six different masks on three datasets.
- RePaint outperformed SOTA methods in 42 of 44 cases. [[Paper]](https://bit.ly/3AwRtLN)

<br>

# Explore the Visual Examples
- Datasets: CelebA-HQ, ImageNet, Places2
- Masks: Random strokes, half image, huge, sparse
- Explore more examples like this in the [[Appendix]](https://bit.ly/3tRVgSJ).


<img width="1556" alt="Denosing Diffusion Inpainting Examples" src="https://user-images.githubusercontent.com/11280511/150864677-0eb482ae-c114-4b0b-b1e0-9be9574da307.png">




<br>

# Code will be released soon

Star this project to get notified.

<p align="center">
  <a href="#"><img src="https://user-images.githubusercontent.com/11280511/150838003-01136487-6671-4cc4-ae4d-bef8e60c605e.png"></a>
</p>

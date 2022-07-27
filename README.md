# RePaint
**Inpainting using Denoising Diffusion Probabilistic Models**


CVPR 2022 [[Paper]](https://bit.ly/3b1ABEb)

[![Denoising_Diffusion_Inpainting_Animation](https://user-images.githubusercontent.com/11280511/150849757-5cd762cb-07a3-46aa-a906-0fe4606eba3b.gif)](#)

## Setup

### 1. Code

```bash
git clone https://github.com/andreas128/RePaint.git
```

### 2. Environment
```bash
pip install numpy torch blobfile tqdm pyYaml pillow    # e.g. torch 1.7.1+cu110.
```

### 3. Download models and data

```bash
pip install --upgrade gdown && bash ./download.sh
```

That downloads the models for ImageNet, CelebA-HQ, and Places2, as well as the face example and example masks.


### 4. Run example
```bash
python test.py --conf_path confs/face_example.yml
```
Find the output in `./log/face_example/inpainted`

*Note: After refactoring the code, we did not reevaluate all experiments.*

<br>

# RePaint fills a missing image part using diffusion models

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
Our noise schedule improves the harmony between the generated and <br> the known part [[4.2 Resampling]](https://bit.ly/3b1ABEb).

<br>

## Details on data

**Which datasets and masks have a ready-to-use config file?**

We provide config files for ImageNet (inet256), CelebA-HQ (c256) and Places2 (p256) for the masks "thin", "thick", "every second line", "super-resolution", "expand" and "half" in [`./confs`](https://github.com/andreas128/RePaint/tree/main/confs). You can use them as shown in the example above.

**How to prepare the test data?**

We use [LaMa](https://github.com/saic-mdal/lama) for validation and testing. Follow their instructions and add the images as specified in the config files. When you download the data using `download.sh`, you can see examples of masks we used.

**How to apply it to other images?**

Copy the config file for the dataset that matches your data best (for faces aligned like CelebA-HQ `_c256`, for diverse images `_inet256`). Then set the [`gt_path`](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/confs/face_example.yml#L70) and [`mask_path`](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/confs/face_example.yml#L71) to where your input is. The masks have the value 255 for known regions and 0 for unknown areas (the ones that get generated).

**How to apply it for other datasets?**

If you work with other data than faces, places or general images, train a model using the [guided-diffusion](https://github.com/openai/guided-diffusion) repository. Note that RePaint is an inference scheme. We do not train or finetune the diffusion model but condition pre-trained models.

## Adapt the code

**How to design a new schedule?**

Fill in your own parameters in this [line](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/guided_diffusion/scheduler.py#L180) to visualize the schedule using `python guided_diffusion/scheduler.py`. Then copy a config file, set your parameters in these [lines](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/confs/face_example.yml#L61-L65) and run the inference using `python test.py --conf_path confs/my_schedule.yml`. 

**How to speed up the inference?**

The following settings are in the [schedule_jump_params](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/confs/face_example.yml#L61) key in the config files. You can visualize them as described above.

- Reduce `t_T`, the total number of steps (without resampling). The lower it is, the more noise gets removed per step.
- Reduce `jump_n_sample` to resample fewer times.
- Apply resampling not from the beginning but only after a specific time by setting `start_resampling`.

## Code overview

- **Schedule:** The list of diffusion times t which will be traversed are obtained in this [line](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L503). e.g. times = [249, 248, 249, 248, 247, 248, 247, 248, 247, 246, ...]
- **Denoise:** Reverse diffusion steps from x<sub>t</sub> (more noise) to a x<sub>t-1</sub> (less noisy) are done below this [line](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L515).
- **Predict:** The model is called [here](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L237) and obtains x<sub>t</sub> and the time t to predict a tensor with 6 channels containing information about the mean and variance of x<sub>t-1</sub>. Then the value range of the variance is adjusted [here](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L252). The mean of x<sub>t-1</sub> is obtained by the weighted sum of the estimated [x<sub>0</sub>](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L270) and x<sub>t</sub> [here](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L189). The obtained mean and variance is used [here](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L402) to sample x<sub>t-1</sub>. (This is the original reverse step from [guided-diffusion](https://github.com/openai/guided-diffusion.git). )
- **Condition:** The known part of the input image needs to have the same amount of noise as the part that the diffusion model generates to join them. The required amount of noise is calculated [here](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L368) and added to the known part [here](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L371). The generated and sampled parts get joined using a maks [here](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L373).
- **Undo:** The forward diffusion steps from x<sub>t-1</sub> to x<sub>t</sub> is done after this [line](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L536). The noise gets added to x<sub>t-1</sub> [here](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L176).

## Issues

**Do you have further questions?**

Please open an [issue](https://github.com/andreas128/RePaint/issues), and we will try to help you.

**Did you find a mistake?**

Please create a pull request. For examply by clicking the pencil button on the top right on the github page.

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

- The inpainting only knows pixels with a stridden access of 2.
- A ratio of 3/4 of the image has to be filled.
- This is equivalent to Super-Resolution with the Nearest Neighbor kernel.

<br>

# RePaint conditions the diffusion model on the known part

- RePaint uses unconditionally trained Denoising Diffusion Probabilistic Models.
- We condition during inference on the given image content.

![Denoising Diffusion Probabilistic Models Inpainting Method](https://user-images.githubusercontent.com/11280511/180631151-59b6674b-bf2c-4501-8307-03c9f5f593ae.gif)

**Intuition of one conditioned denoising step:**
1) **Sample the known part:** Add gaussian noise to the known regions of the image. <br> We obtain a noisy image that follows the denoising process exactly.
2) **Denoise one step:** Denoise the previous image for one step. This generates  <br> content for the unknown region conditioned on the known region.
3) **Join:** Merge the images from both steps.

Details are in Algorithm 1 on Page 5. [[Paper]](https://bit.ly/3b1ABEb)


<br>

# How to harmonize the generated with the known part?

- **Fail:** When using only the algorithm above, the filling is not well harmonized with the known part (n=1).
- **Fix:** When applying the [[4.2 Resampling]](https://bit.ly/3b1ABEb) technique, the images are better harmonized (n>1).

<img width="1577" alt="Diffusion Model Resampling" src="https://user-images.githubusercontent.com/11280511/150822917-737c00b0-b6bb-439d-a5bf-e73238d30990.png">

<br>

# RePaint Fails
- The ImageNet model is biased towards inpainting dogs.
- This is due to the high ratio of dog images in ImageNet.

<img width="1653" alt="RePaint Fails" src="https://user-images.githubusercontent.com/11280511/150853163-b965f59c-5ad4-485b-816e-4391e77b5199.png">

<br>

# User Study State-of-the-Art Comparison

- Outperforms autoregression-based and GAN-based SOTA methods, <br> with 95% significance for all masks except for two inconclusive cases.
- The user study was done for six different masks on three datasets.
- RePaint outperformed SOTA methods in 42 of 44 cases. [[Paper]](https://bit.ly/3b1ABEb)

<br>

# Explore the Visual Examples
- Datasets: CelebA-HQ, ImageNet, Places2
- Masks: Random strokes, half image, huge, sparse
- Explore more examples like this in the [[Appendix]](https://bit.ly/3b1ABEb).


<img width="1556" alt="Denosing Diffusion Inpainting Examples" src="https://user-images.githubusercontent.com/11280511/150864677-0eb482ae-c114-4b0b-b1e0-9be9574da307.png">


<br>


# Acknowledgement

This work was supported by the ETH Zürich Fund (OK), a Huawei Technologies Oy (Finland) project, and an Nvidia GPU grant.

This repository is based on [guided-diffuion](https://github.com/openai/guided-diffusion.git) from OpenAI.

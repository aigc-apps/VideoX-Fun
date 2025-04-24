# CogVideoX FUN v1.1 Report

In CogVideoX-FUN v1.1, we performed additional filtering on the previous dataset, selecting videos with larger motion amplitudes rather than still images in motion, resulting in approximately 0.48 million videos. The model continues to support both image and video prediction, accommodating pixel values from 512x512x49, 768x768x49, 1024x1024x49, and videos with different aspect ratios. We support both image-to-video generation and video-to-video reconstruction.

Additionally, we have released training and prediction code for adding control signals, along with the initial version of the Control model.

Compared to version 1.0, CogVideoX-FUN V1.1 highlights the following features:
- In the 5b model, Noise has been added to the reference images, increasing the motion amplitude of the videos.
- Released training and prediction code for adding control signals, along with the initial version of the Control model.

## Adding Noise to Reference Images

Building on the original CogVideoX-FUN V1.0, we drew upon [CogVideoX](https://github.com/THUDM/CogVideo/) and [SVD](https://github.com/Stability-AI/generative-models) to add Noise upwards to the non-zero reference images to disrupt the original images, aiming for greater motion amplitude.

In our 5b model, Noise has been added, while the 2b model only performed fine-tuning with new data. This is because, after attempting to add Noise in the 2b model, the generated videos exhibited excessive motion amplitude, leading to deformation and damaging the output. The 5b model, due to its stronger generative capabilities, maintains relatively stable outputs during motion.

Furthermore, the prompt words significantly influence the generation results, so please describe the actions in detail to increase dynamism. If unsure how to write positive prompts, you can use phrases like "smooth motion" or "in the wind" to enhance dynamism. Additionally, it is advisable to avoid using dynamic terms like "motion" in negative prompts.

## Adding Control Signals to CogVideoX-FUN

On the basis of the original CogVideoX-FUN V1.0, we replaced the original mask signal with Pose control signals. The control signals are encoded using VAE and used as Guidance, along with latent data entering the patch processing flow.

We filtered the 0.48 million dataset, selecting around 20,000 videos and images containing portraits for pose extraction, which served as condition control signals for training. 

During the training process, the videos are scaled according to different Token lengths. The entire training process is divided into two phases, with each phase comprising 13,312 (corresponding to 512x512x49 videos) and 53,248 (corresponding to 1024x1024x49 videos).

Taking CogVideoX-Fun-V1.1-5b-Pose as an example:
- In the 13312 phase, the batch size is 128, with 2.4k training steps.
- In the 53248 phase, the batch size is 128, with 1.2k training steps.

The working principle diagram is shown below:
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pipeline_control.jpg" alt="ui" style="zoom:50%;" />

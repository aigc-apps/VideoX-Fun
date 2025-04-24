# CogVideoX FUN v1 Report
In CogVideoX-FUN, we trained on approximately 1.2 million data points based on CogVideoX, supporting image and video predictions. It accommodates pixel values for video generation across different resolutions of 512x512x49, 768x768x49, and 1024x1024x49, as well as videos with different aspect ratios. Moreover, we support the generation of videos from images and the reconstruction of videos from other videos.

Compared to CogVideoX, CogVideoX FUN also highlights the following features:
- Introduction of the InPaint model, enabling the generation of videos from images with specified starting and ending images.
- Training the model based on token lengths. This allows for the implementation of various sizes and resolutions within the same model.

## InPaint Model
We used [CogVideoX](https://github.com/THUDM/CogVideo/) as the foundational structure, referencing [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) for the model training to generate videos from images. 

During video generation, the **reference video** is encoded using VAE, with the **black area in the above image representing the part to be reconstructed, and the white area representing the start image**. This is stacked with noise latents and input into the Transformer for video generation. We perform 3D resizing on the **masked area**, directly resizing it to fit the canvas size of the video that needs reconstruction. 

Then, we concatenate the latent, the encoded reference video, and the masked area, inputting them into DiT for noise prediction to obtain the final video. 
The pipeline structure of CogVideoX FUN is as follows:
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/pipeline.jpg" alt="ui" style="zoom:50%;" />

## Token Length-Based Model Training
We collected approximately 1.2 million high-quality data for the training of CogVideoX-Fun. During the training, we resized the videos based on different token lengths. The entire training process is divided into three phases, with each phase corresponding to 13312 (for 512x512x49 videos), 29952 (for 768x768x49 videos), and 53248 (for 1024x1024x49 videos).

Taking CogVideoX-Fun-2B as an example:
- In the 13312 phase, the batch size is 128 with 7k training steps.
- In the 29952 phase, the batch size is 256 with 6.5k training steps.
- In the 53248 phase, the batch size is 128 with 5k training steps.

During training, we combined high and low resolutions, enabling the model to support video generation from any resolution between 512 and 1280. For example, with a token length of 13312:
- At a resolution of 512x512, the number of video frames is 49.
- At a resolution of 768x768, the number of video frames is 21.
- At a resolution of 1024x1024, the number of video frames is 9.

These resolutions and corresponding lengths were mixed for training, allowing the model to generate videos at different resolutions.

## Resize 3D Embedding
In adapting CogVideoX-2B to the CogVideoX-Fun framework, it was found that the source code obtains 3D embeddings in a truncated manner. This approach only accommodates a single resolution; when the resolution changes, the embedding should also change.
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/PE_Interpolation.jpg" alt="ui" style="zoom:50%;" />

Referencing Pixart-Sigma, the above image is from the Pixart-Sigma paper. We used Positional Embeddings Interpolation (PE Interpolation) to resize 3D embeddings. PE Interpolation is more conducive to convergence than directly generating cosine and sine embeddings for different resolutions.

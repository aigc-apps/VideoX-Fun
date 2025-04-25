# Wan Fun v1.1 Report

In Wan-Fun v1.1, we updated six models: the 14B Inpaint model, Control model, and Control-Camera model; as well as the 1.3B Inpaint model, Control model, and Control-Camera model.

Compared to the previous version, the Inpaint model has been trained with a larger batch size, resulting in more stable performance. The Control model now includes a reference image model to achieve effects similar to Animate Anyone. While retaining its original functionality, it can also accept both a reference image and a control video as inputs for generation. Finally, we provide a camera control model that supports pan-and-tilt movements (left, right, up, down).

Additionally, we have released training and inference code for adding reference control signals, as well as training and inference code for adding camera control signals.

Compared to V1.0, Wan Fun V1.1 highlights the following features:

- A more stable Inpaint model.
- On top of the original control scheme, we’ve implemented a new control approach combining reference images and control videos.
- Added support for a camera control model.

## Implementation of Reference Image + Control Video
In Wan-Fun V1.0, we already supported multiple control signals such as Canny, Depth, Pose, and MLSD, and implemented two control schemes: initial-image plus trajectory control, and control-video-guided generation.

To further enhance the usability of the control model, we developed a new control scheme that combines reference images with control videos, akin to Animate Anyone. This feature takes a reference image as input and generates output based on control signals like Openpose (though it is not limited to Openpose—Depth signals also yield impressive results). Previously, methods like Unimate or Animate Anyone typically required the skeleton of the reference image to closely align with the control video. However, in Wan-Fun V1.1 Control, even if there is some misalignment, the system still produces acceptable results, though better alignment naturally leads to higher similarity.

We encode the reference image using VAE, then tile the latent representation and concatenate the tiled features with the video features for generation. To ensure that this does not interfere with the model's original functionality, during training, we randomly initialize the reference image latent as all zeros to simulate cases where no reference image is provided. The overall workflow of the model is shown in the figure below:

![Control_Ref](https://github.com/user-attachments/assets/8987a3b5-e691-4c49-a83c-10cad0fe13f3)

## Camera Control Model
Building upon Wan-Fun V1.0, we now support additional camera information inputs for camera control.

Inspired by [CameraCtrl](https://github.com/hehao13/CameraCtrl) and [EasyAnimate](https://github.com/aigc-apps/EasyAnimate), instead of directly resizing inputs like EasyAnimate does to input camera trajectories, we first use PixelUnshuffle to convert temporal information into channel information. Then, using an adapter mechanism, we transform the camera trajectory into high-level semantic information before adding it to the video features post-Conv. This allows us to achieve precise control over the camera lens movement.

The overall framework of the model is shown in the figure below:

![Control_Camera](https://github.com/user-attachments/assets/0fdb129f-7c74-48e6-9fbd-9fef23ef446e)
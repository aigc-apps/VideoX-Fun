# Wan Fun v1.1 Report

在Wan-Fun v1.1中，我们更新了6个模型，分别是：14B的Inpaint模型，Control模型、Control-Camera模型；1.3B的Inpaint模型，Control模型、Control-Camera模型。

相比于上一个版本，Inpaint模型经过了更大batch size的训练，模型效果的稳定性更优秀；Control模型则新增加了一个参考图模型，以实现类似于Animate Anyone的效果，在保留之前功能的基础上，我们可以同时传入参考图片和控制视频，以实现生成；最后我们提供了镜头控制模型，可以实现上下左右的镜头控制。

另外，我们还发布了添加参考控制信号的训练代码与预测代码，添加镜头控制信号的训练代码和预测代码。

对比V1.0版本，Wan Fun V1.1突出了以下功能：

- 更为稳定的Inpaint模型。
- 在原控制方案的基础上，实现了参考图加上控制视频的控制方案。
- 实现了镜头控制模型。

## 参考图加上控制视频的实现
在原本Wan-Fun V1.0的中，我们已经支持了多种控制信号，如Canny、Depth、Pose、MLSD；实现了两种控制方案，如首图+轨迹控制、控制视频指导生成。

为了进一步提高控制模型的可用性，我们进一步开发了参考图加上控制视频的控制方案，该功能animate anyone，输入一张参考图片，然后根据Openpose（并不局限于Openpose，Depth信号也有非常亮眼的效果）之类的控制实现生成。此前，Unimate、animate anyone之类的方案一般要求参考图和控制视频骨架基本对齐。在Wan-Fun V1.1 Control中，对齐可以有更好的相似度，不对齐也可以有一定的参考生成效果。

我们将参考图使用VAE Encode之后，将latent平铺，然后将平铺后的特征与视频特征进行concat实现生成，为了不影响模型的原功能，我们在训练中随机将参考图latent初始化为全0，代表没有参考图输入，整体模型的工作框架如图所示：

![Control_Ref](https://github.com/user-attachments/assets/8987a3b5-e691-4c49-a83c-10cad0fe13f3)

## 镜头控制模型
在原本Wan-Fun V1.0的基础上，我们支持进一步输入Camera信息，以进行镜头控制。

参考[CameraCtrl](https://github.com/hehao13/CameraCtrl)与[EasyAnimate](https://github.com/aigc-apps/EasyAnimate)，我们没有选择类似于EasyAnimate那种直接Resize的方式输入相机镜头的控制轨迹，而是先使用PixelUnshuffle将时序信息转换成通道信息，然后使用Adapter的方式，将相机镜头的轨迹转换成高层语义信息后再与Conv in后的视频特征相加，从而实现了视频镜头的控制。

整体模型的工作框架如图所示：

![Control_Camera](https://github.com/user-attachments/assets/0fdb129f-7c74-48e6-9fbd-9fef23ef446e)
# CogVideoX FUN v1 Report

在CogVideoX-FUN中，我们基于CogVideoX在大约1.2m的数据上进行了训练，支持图片与视频预测，支持像素值从512x512x49、768x768x49、1024x1024x49与不同纵横比的视频生成。另外，我们支持图像到视频的生成与视频到视频的重建。

对比与CogVideoX，CogVideoX FUN还突出了以下功能：

- 引入InPaint模型，实现图生视频功能，可以通过首尾图指定视频生成。
- 基于Token长度的模型训练。达成不同大小多分辨率在同一模型中的实现。

## InPaint模型
我们以[CogVideoX](https://github.com/THUDM/CogVideo/)作为基础结构，参考[EasyAnimate](https://github.com/aigc-apps/EasyAnimate)进行图生视频的模型训练。

在进行视频生成的时候，将**参考视频**使用VAE进行encode，**上图黑色的部分代表需要重建的部分，白色的部分代表首图**，与噪声Latents一起堆叠后输入到Transformer中进行视频生成。

我们对**被Mask的区域**进行3D Resize，直接Resize到需要重建的视频的画布大小。

然后将Latent、Encode后的参考视频、被Mask的区域，concat后输入到DiT中进行噪声预测。获得最终的视频。

CogVideoX FUN的Pipeline结构如下：
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/pipeline.jpg" alt="ui" style="zoom:50%;" />

## 基于Token长度的模型训练
我们收集了大约高质量的1.2m数据进行CogVideoX-Fun的训练。

在进行训练时，我们根据不同Token长度，对视频进行缩放后进行训练。整个训练过程分为三个阶段，每个阶段的13312（对应512x512x49的视频），29952（对应768x768x49的视频），53248（对应1024x1024x49的视频）。

以CogVideoX-Fun-2B为例子，其中：
- 13312阶段，Batch size为128，训练步数为7k
- 29952阶段，Batch size为256，训练步数为6.5k。
- 53248阶段，Batch size为128，训练步数为5k。

训练时我们采用高低分辨率结合训练，因此模型支持从512到1280任意分辨率的视频生成，以13312 token长度为例：
- 在512x512分辨率下，视频帧数为49；
- 在768x768分辨率下，视频帧数为21；
- 在1024x1024分辨率下，视频帧数为9；
这些分辨率与对应长度混合训练，模型可以完成不同大小分辨率的视频生成。

## Resize 3D Embedding
在适配CogVideoX-2B到CogVideoX-Fun框架的途中，发现源码是以截断的方式去得到3D Embedding的，这样的方式只能适配单一分辨率，当分辨率发生变化时，Embedding也应当发生变化。

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/PE_Interpolation.jpg" alt="ui" style="zoom:50%;" />

参考Pixart-Sigma，上图来自于Pixart-Sigma论文，我们采用Positional Embeddings Interpolation（PE Interpolation）对3D embedding进行Resize，PE Interpolation相比于直接生成不同分辨率的Cos Sin Embedding更易收敛。
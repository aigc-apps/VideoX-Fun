# CogVideoX FUN v1.1 Report

在CogVideoX-FUN v1.1中，我们在之前的数据集中再次做了筛选，选出其中动作幅度较大，而不是静止画面移动的视频，数量大约为0.48m。模型依然支持图片与视频预测，支持像素值从512x512x49、768x768x49、1024x1024x49与不同纵横比的视频生成。我们支持图像到视频的生成与视频到视频的重建。

另外，我们还发布了添加控制信号的训练代码与预测代码，并发布了初版的Control模型。

对比V1.0版本，CogVideoX-FUN V1.1突出了以下功能：

- 在5b模型中，给参考图片添加了Noise，增加了视频的运动幅度。
- 发布了添加控制信号的训练代码与预测代码，并发布了初版的Control模型。

## 参考图片添加Noise
在原本CogVideoX-FUN V1.0的基础上，我们参考[CogVideoX](https://github.com/THUDM/CogVideo/)和[SVD](https://github.com/Stability-AI/generative-models)，在非0的参考图向上添加Noise以破环原图，追求更大的运动幅度。

我们5b模型中添加了Noise，2b模型仅使用了新数据进行了finetune，因为我们在2b模型中尝试添加Noise之后，生成的视频运动幅度过大导致结果变形，破坏了生成结果，而5b模型因为更为的强大生成能力，在运动中也保持了较为稳定的输出。

另外，提示词对生成结果影响较大，请尽量描写动作以增加动态性。如果不知道怎么写正向提示词，可以使用smooth motion or in the wind来增加动态性。并且尽量避免在负向提示词中出现motion等表示动态的词汇。

## 添加控制信号的CogVideoX-Fun
在原本CogVideoX-FUN V1.0的基础上，我们使用Pose控制信号替代了原本的mask信号，将控制信号使用VAE编码后作为Guidance与latent一起进入patch流程，

我们在0.48m数据中进行了筛选，选择出大约20000包含人像的视频与图片进行pose提取，作为condition控制信号进行训练。

在进行训练时，我们根据不同Token长度，对视频进行缩放后进行训练。整个训练过程分为两个阶段，每个阶段的13312（对应512x512x49的视频），53248（对应1024x1024x49的视频）。

以CogVideoX-Fun-V1.1-5b-Pose为例子，其中：
- 13312阶段，Batch size为128，训练步数为2.4k
- 53248阶段，Batch size为128，训练步数为1.2k。

工作原理图如下：
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pipeline_control.jpg" alt="ui" style="zoom:50%;" />

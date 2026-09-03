# VideoX-Fun

😊 ようこそ！

CogVideoX-Fun:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/CogVideoX-Fun-5b)

Wan-Fun:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/Wan2.1-Fun-1.3B-InP)

[English](./README.md) | [简体中文](./README_zh-CN.md) | 日本語

# 目次
- [一、紹介](#一紹介)
- [二、クイックスタートと使用](#二クイックスタートと使用)
  - [1. 環境準備](#1-環境準備)
  - [2. 推論生成](#2-推論生成)
  - [3. モデルのトレーニング](#3-モデルのトレーニング)
- [三、サポート済みモデル](#三サポート済みモデル)
- [四、ビデオ作品](#四ビデオ作品)
- [五、参考文献](#五参考文献)
- [六、引用](#六引用)
- [七、制限とリスク](#七制限とリスク)
- [八、ライセンス](#八ライセンス)

# 一、紹介
VideoX-Funはビデオ生成のパイプラインであり、AI画像やビデオの生成、Diffusion TransformerのベースラインモデルとLoraモデルのトレーニングに使用できます。我々は、すでに学習済みのベースラインモデルから直接予測を行い、異なる解像度、秒数、FPSのビデオを生成することをサポートしています。また、ユーザーが独自のベースラインモデルやLoraモデルをトレーニングし、特定のスタイル変換を行うこともサポートしています。

新機能：
- Wan 2.2シリーズモデル、Wan-VACE制御モデル、Fantasy Talkingデジタルヒューマンモデル、Qwen-Image、Flux画像生成モデルなどのサポートを追加しました。[2025.10.16]
- Wan2.1-Fun-V1.1バージョンを更新：14Bと1.3BモデルのControl＋参照画像モデルをサポート、カメラ制御にも対応。さらに、Inpaintモデルを再訓練し、性能が向上しました。[2025.04.25]
- Wan2.1-Fun-V1.0の更新：14Bおよび1.3BのI2V（画像からビデオ）モデルとControlモデルをサポートし、開始フレームと終了フレームの予測に対応。[2025.03.26]
- CogVideoX-Fun-V1.5の更新：I2Vモデルと関連するトレーニング・予測コードをアップロード。[2024.12.16]
- 報酬Loraのサポート：報酬逆伝播技術を使用してLoraをトレーニングし、生成された動画を最適化し、人間の好みによりよく一致させる。[詳細情報](scripts/README_TRAIN_REWARD.md)。新しいバージョンの制御モデルでは、Canny、Depth、Pose、MLSDなどの異なる制御条件に対応。[2024.11.21]
- diffusersのサポート：CogVideoX-Fun Controlがdiffusersでサポートされるようになりました。[a-r-r-o-w](https://github.com/a-r-r-o-w)がこの[PR](https://github.com/huggingface/diffusers/pull/9671)でサポートを提供してくれたことに感謝します。詳細は[ドキュメント](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox)をご覧ください。[2024.10.16]
- CogVideoX-Fun-V1.1の更新：i2vモデルを再トレーニングし、Noiseを追加して動画の動きの範囲を拡大。制御モデルのトレーニングコードとControlモデルをアップロード。[2024.09.29]
- CogVideoX-Fun-V1.0の更新：コードを作成！WindowsとLinuxに対応しました。2Bおよび5Bモデルでの最大256x256x49から1024x1024x49までの任意の解像度の動画生成をサポート。[2024.09.18]

機能：
- [データ前処理](#data-preprocess)
- [DiTのトレーニング](#dit-train)
- [ビデオ生成](#video-gen)

私たちのUIインターフェースは次のとおりです：
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/ui.jpg)

# 二、クイックスタートと使用

<a id="quick-start"></a>

## 1. 環境準備

### 1.1 クラウド使用: AliyunDSW

DSWには無料のGPU時間があり、ユーザーは一度申請でき、申請後3か月間有効です。

Aliyunは[Freetier](https://free.aliyun.com/?product=9602825&crowd=enterprise&spm=5176.28055625.J_5831864660.1.e939154aRgha4e&scm=20140722.M_9974135.P_110.MO_1806-ID_9974135-MID_9974135-CID_30683-ST_8512-V_1)で無料のGPU時間を提供しています。取得してAliyun PAI-DSWで使用し、5分以内にCogVideoX-Funを開始できます！

[![DSW Notebook](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/dsw.png)](https://gallery.pai-ml.com/#/preview/deepLearning/cv/cogvideox_fun)

### 1.2 ローカル依存のインストール

以下の環境でこのライブラリの実行を確認しています：

Windowsの詳細：
- OS: Windows 10
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU： Nvidia-3060 12G & Nvidia-3090 24G

Linuxの詳細：
- OS: Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

重みを保存するために約60GBのディスクスペースが必要です。確認してください！

### 1.3 Dockerの使用

Dockerを使用する場合、マシンにグラフィックスカードドライバとCUDA環境が正しくインストールされていることを確認してください。

次のコマンドをこの方法で実行します：

```
# イメージをプル
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# イメージに入る
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# コードをクローン
git clone https://github.com/aigc-apps/VideoX-Fun.git

# VideoX-Funのディレクトリに入る
cd VideoX-Fun

# 重みをダウンロード
mkdir models/Diffusion_Transformer
mkdir models/Personalized_Model

# Please use the hugginface link or modelscope link to download the model.
# CogVideoX-Fun
# https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP
# https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-InP

# Wan
# https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-14B-InP
# https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP
```

### 1.4 重みの配置

[重み](#三サポート済みモデル)を指定されたパスに配置することをお勧めします：

**ComfyUIを通じて**:
モデルをComfyUIの重みフォルダ `ComfyUI/models/Fun_Models/` に入れます：
```
📦 ComfyUI/
├── 📂 models/
│   └── 📂 Fun_Models/
│       ├── 📂 CogVideoX-Fun-V1.1-2b-InP/
│       ├── 📂 CogVideoX-Fun-V1.1-5b-InP/
│       ├── 📂 Wan2.1-Fun-V1.1-14B-InP
│       └── 📂 Wan2.1-Fun-V1.1-1.3B-InP/
```

**独自のpythonファイルまたはUIインターフェースを実行**:
```
📦 models/
├── 📂 Diffusion_Transformer/
│   ├── 📂 CogVideoX-Fun-V1.1-2b-InP/
│   ├── 📂 CogVideoX-Fun-V1.1-5b-InP/
│   ├── 📂 Wan2.1-Fun-V1.1-14B-InP
│   └── 📂 Wan2.1-Fun-V1.1-1.3B-InP/
├── 📂 Personalized_Model/
│   └── あなたのトレーニング済みのトランスフォーマーモデル / あなたのトレーニング済みのLoraモデル（UIロード用）
```

## 2. 推論生成

<a id="video-gen"></a>

ビデオモデルと画像モデルの推論入口は完全に一致しており、`examples/{model_name}/`下のスクリプトまたはUIから実行します。

### 2.1 入口の選択

| 使用入口 | 適用シーン | 設定粒度 |
|--|--|--|
| Pythonファイル | バッチ生成、スクリプト内でパラメータを調整 | 全パラメータ |
| WebUI | 対話的な体験、モデルの迅速な切り替え | よく使うパラメータのみ |
| ComfyUI | 既存のComfyUIワークフロー | ノードパラメータ |

表：推論入口の選択

### 2.2 顕存節約方案

Wan2.1のパラメータが非常に大きいため、GPUメモリを節約し、コンシューマー向けGPUに適応させる必要があります。各予測ファイルには`GPU_memory_mode`を提供しており、`model_cpu_offload`、`model_cpu_offload_and_qfloat8`、`sequential_cpu_offload`の中から選択できます。この方法はCogVideoX-Funの生成にも適用されます。

- `model_cpu_offload`: モデル全体が使用後にCPUに移動し、一部のGPUメモリを節約します。
- `model_cpu_offload_and_qfloat8`: モデル全体が使用後にCPUに移動し、Transformerモデルに対してfloat8の量子化を行い、より多くのGPUメモリを節約します。
- `sequential_cpu_offload`: モデルの各層が使用後にCPUに移動します。速度は遅くなりますが、大量のGPUメモリを節約します。

`qfloat8`はモデルの性能を部分的に低下させる可能性がありますが、より多くのGPUメモリを節約できます。十分なGPUメモリがある場合は、`model_cpu_offload`の使用をお勧めします。

### 2.3 Pythonファイルから

##### i. 単一GPUでの推論:

- ステップ1: 対応する[重み](#三サポート済みモデル)をダウンロードし、`models`フォルダに配置します。
- ステップ2: 異なる重みと予測目標に基づいて、異なるファイルを使用して予測を行います。現在、このライブラリはCogVideoX-Fun、Wan2.1、およびWan2.1-Funをサポートしています。`examples`フォルダ内のフォルダ名で区別され、異なるモデルがサポートする機能が異なりますので、状況に応じて区別してください。以下はCogVideoX-Funを例として説明します。
  - テキストからビデオ:
    - `examples/cogvideox_fun/predict_t2v.py`ファイルで`prompt`、`neg_prompt`、`guidance_scale`、`seed`を変更します。
    - 次に、`examples/cogvideox_fun/predict_t2v.py`ファイルを実行し、結果が生成されるのを待ちます。結果は`samples/cogvideox-fun-videos`フォルダに保存されます。
  - 画像からビデオ:
    - `examples/cogvideox_fun/predict_i2v.py`ファイルで`validation_image_start`、`validation_image_end`、`prompt`、`neg_prompt`、`guidance_scale`、`seed`を変更します。
    - `validation_image_start`はビデオの開始画像、`validation_image_end`はビデオの終了画像です。
    - 次に、`examples/cogvideox_fun/predict_i2v.py`ファイルを実行し、結果が生成されるのを待ちます。結果は`samples/cogvideox-fun-videos_i2v`フォルダに保存されます。
  - ビデオからビデオ:
    - `examples/cogvideox_fun/predict_v2v.py`ファイルで`validation_video`、`validation_image_end`、`prompt`、`neg_prompt`、`guidance_scale`、`seed`を変更します。
    - `validation_video`はビデオ生成のための参照ビデオです。以下のデモビデオを使用して実行できます：[デモビデオ](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/play_guitar.mp4)
    - 次に、`examples/cogvideox_fun/predict_v2v.py`ファイルを実行し、結果が生成されるのを待ちます。結果は`samples/cogvideox-fun-videos_v2v`フォルダに保存されます。
  - 通常の制御付きビデオ生成（Canny、Pose、Depthなど）:
    - `examples/cogvideox_fun/predict_v2v_control.py`ファイルで`control_video`、`validation_image_end`、`prompt`、`neg_prompt`、`guidance_scale`、`seed`を変更します。
    - `control_video`は、Canny、Pose、Depthなどの演算子で抽出された制御用ビデオです。以下のデモビデオを使用して実行できます：[デモビデオ](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4)
    - 次に、`examples/cogvideox_fun/predict_v2v_control.py`ファイルを実行し、結果が生成されるのを待ちます。結果は`samples/cogvideox-fun-videos_v2v_control`フォルダに保存されます。
- ステップ3: 自分でトレーニングした他のバックボーンやLoraを組み合わせたい場合は、必要に応じて`examples/{model_name}/predict_t2v.py`や`examples/{model_name}/predict_i2v.py`、`lora_path`を修正します。

##### ii. 複数GPUでの推論:
多カードでの推論を行う際は、xfuserリポジトリのインストールに注意してください。xfuser==0.4.2 と yunchang==0.6.2 のインストールが推奨されます。
```
pip install xfuser==0.4.2 --progress-bar off -i https://mirrors.aliyun.com/pypi/simple/
pip install yunchang==0.6.2 --progress-bar off -i https://mirrors.aliyun.com/pypi/simple/
```

`ulysses_degree` と `ring_degree` の積が使用する GPU 数と一致することを確認してください。たとえば、8つのGPUを使用する場合、`ulysses_degree=2` と `ring_degree=4`、または `ulysses_degree=4` と `ring_degree=2` を設定することができます。

- `ulysses_degree` はヘッド（head）に分割した後の並列化を行います。
- `ring_degree` はシーケンスに分割した後の並列化を行います。

`ring_degree` は `ulysses_degree` よりも通信コストが高いため、これらのパラメータを設定する際には、シーケンス長とモデルのヘッド数を考慮する必要があります。

8GPUでの並列推論を例に挙げます：

- **Wan2.1-Fun-V1.1-14B-InP** はヘッド数が40あります。この場合、`ulysses_degree` は40で割り切れる値（例：2, 4, 8など）に設定する必要があります。したがって、8GPUを使用して並列推論を行う場合、`ulysses_degree=8` と `ring_degree=1` を設定できます。

- **Wan2.1-Fun-V1.1-1.3B-InP** はヘッド数が12あります。この場合、`ulysses_degree` は12で割り切れる値（例：2, 4など）に設定する必要があります。したがって、8GPUを使用して並列推論を行う場合、`ulysses_degree=4` と `ring_degree=2` を設定できます。

パラメータの設定が完了したら、以下のコマンドで並列推論を実行してください：

```sh
torchrun --nproc-per-node=8 examples/wan2.1_fun/predict_t2v.py
```

### 2.4 UIインターフェースから

WebUIは、テキストからビデオ、画像からビデオ、ビデオからビデオ、および通常の制御付きビデオ生成（Canny、Pose、Depthなど）をサポートします。現在、このライブラリはCogVideoX-Fun、Wan2.1、およびWan2.1-Funをサポートしており、`examples`フォルダ内のフォルダ名で区別されています。異なるモデルがサポートする機能が異なるため、状況に応じて区別してください。以下はCogVideoX-Funを例として説明します。

- ステップ1: 対応する[重み](#三サポート済みモデル)をダウンロードし、`models`フォルダに配置します。
- ステップ2: `examples/cogvideox_fun/app.py`ファイルを実行し、Gradioページに入ります。
- ステップ3: ページ上で生成モデルを選択し、`prompt`、`neg_prompt`、`guidance_scale`、`seed`などを入力し、「生成」をクリックして結果が生成されるのを待ちます。結果は`sample`フォルダに保存されます。

### 2.5 ComfyUIから

詳細は[ComfyUI README](comfyui/README.md)をご覧ください。


## 3. モデルのトレーニング

完全なモデルトレーニングパイプラインは、データ前処理とVideo DiTトレーニングで構成されます。

### 3.1 データ前処理

<a id="data-preprocess"></a>
各モデルの訓練ドキュメントは`scripts/{model_name}/`下に統一されています。詳細は[3.3 各モデルの訓練ドキュメント](#33-各モデルの訓練ドキュメント)を参照してください。

長いビデオのセグメンテーション、クリーニング、説明のための完全なデータ前処理リンクは、ビデオキャプションセクションの[README](videox_fun/video_caption/README.md)を参照してください。

テキストから画像およびビデオ生成モデルをトレーニングしたい場合。この形式でデータセットを配置する必要があります。

```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

json_of_internal_datasets.jsonは標準のJSONファイルです。json内のfile_pathは相対パスとして設定できます。以下のように：
```json
[
    {
      "file_path": "train/00000001.mp4",
      "text": "スーツとサングラスを着た若い男性のグループが街の通りを歩いている。",
      "type": "video"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "スーツとサングラスを着た若い男性のグループが街の通りを歩いている。",
      "type": "image"
    },
    .....
]
```

次のように絶対パスとして設定することもできます：
```json
[
    {
      "file_path": "/mnt/data/videos/00000001.mp4",
      "text": "スーツとサングラスを着た若い男性のグループが街の通りを歩いている。",
      "type": "video"
    },
    {
      "file_path": "/mnt/data/train/00000001.jpg",
      "text": "スーツとサングラスを着た若い男性のグループが街の通りを歩いている。",
      "type": "image"
    },
    .....
]
```

### 3.2 Video DiTのトレーニング

<a id="dit-train"></a>
各モデルの訓練スクリプトと起動shは`scripts/{model_name}/`下にあり、shの名称はタスクによって異なります（例：`train.sh`、`train_lora.sh`、`train_control.sh`、`train_control_distill.sh`など）。ディレクトリ内の実際のファイルを基準としてください。

データ前処理時にデータ形式が相対パスの場合、```scripts/{model_name}/train.sh```を次のように設定します。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"
```

データ形式が絶対パスの場合、同じスクリプトで次のように設定します（このとき`DATASET_NAME`は空にし、データセットディレクトリのプレフィックスを連結しません）。
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

最後に対応するスクリプトを実行します。
```sh
sh scripts/{model_name}/train.sh
```

### 3.3 各モデルの訓練ドキュメント

パラメータ設定の詳細について、各モデルの訓練ドキュメントは`scripts/{model_name}/`下に統一されています。

| モデル | ベーストレーニング | LoRAトレーニング | その他 |
|--|--|--|--|
| Wan2.1-Fun | [EN](scripts/wan2.1_fun/README_TRAIN.md) / [ZH](scripts/wan2.1_fun/README_TRAIN_zh-CN.md) | [EN](scripts/wan2.1_fun/README_TRAIN_LORA.md) / [ZH](scripts/wan2.1_fun/README_TRAIN_LORA_zh-CN.md) | [Control ZH](scripts/wan2.1_fun/README_TRAIN_CONTROL_zh-CN.md)、[Reward LoRA](scripts/wan2.1_fun/README_TRAIN_REWARD.md) |
| Wan2.2 | [EN](scripts/wan2.2/README_TRAIN.md) / [ZH](scripts/wan2.2/README_TRAIN_zh-CN.md) | [EN](scripts/wan2.2/README_TRAIN_LORA.md) / [ZH](scripts/wan2.2/README_TRAIN_LORA_zh-CN.md) | [Distill ZH](scripts/wan2.2/README_TRAIN_DISTILL_zh-CN.md)、[S2V](scripts/wan2.2/README_TRAIN_S2V.md)、[Animate](scripts/wan2.2/README_TRAIN_ANIMATE.md) |
| Wan2.2-Fun | [EN](scripts/wan2.2_fun/README_TRAIN.md) / [ZH](scripts/wan2.2_fun/README_TRAIN_zh-CN.md) | [EN](scripts/wan2.2_fun/README_TRAIN_LORA.md) / [ZH](scripts/wan2.2_fun/README_TRAIN_LORA_zh-CN.md) | [Control LoRA ZH](scripts/wan2.2_fun/README_TRAIN_CONTROL_LORA_zh-CN.md) |
| CogVideoX-Fun | [EN](scripts/cogvideox_fun/README_TRAIN.md) / [ZH](scripts/cogvideox_fun/README_TRAIN_zh-CN.md) | [EN](scripts/cogvideox_fun/README_TRAIN_LORA.md) / [ZH](scripts/cogvideox_fun/README_TRAIN_LORA_zh-CN.md) | [Control ZH](scripts/cogvideox_fun/README_TRAIN_CONTROL_zh-CN.md)、[Reward LoRA](scripts/cogvideox_fun/README_TRAIN_REWARD.md) |
| Qwen-Image | [EN](scripts/qwenimage/README_TRAIN.md) / [ZH](scripts/qwenimage/README_TRAIN_zh-CN.md) | [EN](scripts/qwenimage/README_TRAIN_LORA.md) / [ZH](scripts/qwenimage/README_TRAIN_LORA_zh-CN.md) | [Edit ZH](scripts/qwenimage/README_TRAIN_EDIT_zh-CN.md) |
| Z-Image | [EN](scripts/z_image/README_TRAIN.md) / [ZH](scripts/z_image/README_TRAIN_zh-CN.md) | [EN](scripts/z_image/README_TRAIN_LORA.md) / [ZH](scripts/z_image/README_TRAIN_LORA_zh-CN.md) | [GRPO LoRA](scripts/z_image/README_TRAIN_GRPO_LORA.md) |

その他のモデルも同様に、対応する`scripts/{model_name}/`下のREADMEを参照してください。

# 三、サポート済みモデル

下表は、現在サポートされているモデル系列と重みをまとめたものです。ビデオモデルと画像モデルは同じ推論・訓練インターフェースを共有しています。各行は1つのモデル系列を表し、第4列は4列のHTML埋め込みテーブル（重み、Hugging Face、ModelScope、説明）です。🤗 は Hugging Face、🤖 は ModelScope（中国国内ネットワーク向け）、`-` は該当チャネルに対応リポジトリがないか、ログイン認証が必要なことを示します。各モデルの訓練ドキュメントについては[3.3 各モデルの訓練ドキュメント](#33-各モデルの訓練ドキュメント)を参照してください。

| モデル系列 | モダリティ | サポートタスク | 重み / ダウンロード / 説明 |
|--|--|--|--|

| Wan2.2-Fun | ビデオ | 本プロジェクトがWan2.2で訓練した系列。テキスト/画像から動画、首尾画像、制御生成、カメラ制御をカバー | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-Fun-A14B-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-InP">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-Fun-14Bのテキスト・画像から動画を生成するモデルの重み。複数の解像度で学習されており、動画の最初と最後のフレームの予測をサポートしています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-Fun-A14B-Control</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-Fun-14Bの動画制御用重み。Canny、Depth、Pose、MLSDなどのさまざまな制御条件に対応しており、軌跡制御もサポートしています。512、768、1024の複数解像度での動画生成が可能で、81フレーム、16fpsで学習されています。多言語対応の予測もサポートしています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-Fun-A14B-Control-Camera</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control-Camera">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control-Camera">🤖</a></td><td valign="top" style="padding:2px 0;">14B Controlにカメラモーション制御を追加</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-Fun-5B-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-InP">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-Fun-5B テキストから動画生成用の重み。121フレーム、24 FPSで学習され、先頭/末尾フレーム予測をサポート。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-Fun-5B-Control</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-Control">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-Fun-5B 動画制御用重み。Canny、Depth、Pose、MLSDなどの制御条件や軌道制御をサポート。121フレーム、24 FPSで学習され、多言語予測に対応。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-Fun-5B-Control-Camera</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control-Camera">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-Control-Camera">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-Fun-5B カメラレンズ制御用重み。121フレーム、24 FPSで学習され、多言語予測に対応。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-Fun-Reward-LoRAs</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.2-Fun-Reward-LoRAs">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-Fun生成動画を報酬逆伝播で最適化するReward LoRA集合</td></tr></table> |
| Wan2.2-VACE-Fun | ビデオ | 本プロジェクトがVACE方式で訓練した系列。制御生成と主題参照をカバー | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-VACE-Fun-A14B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.2-VACE-Fun-A14B">🤖</a></td><td valign="top" style="padding:2px 0;">VACE方式でトレーニングされたWan2.2の制御ウェイト（ベースモデルはWan2.2-T2V-A14B）。Canny、Depth、Pose、MLSD、軌道制御などの異なる制御条件をサポートします。対象を指定して動画生成が可能です。多解像度（512、768、1024）の動画予測をサポートし、81フレームで16FPSでトレーニングされています。多言語予測にも対応しています。</td></tr></table> |
| Wan2.2 | ビデオ | Wan公式重み。テキスト/画像から動画、音声駆動、キャラクターアニメーションをカバー。Wan2.2-Fun系列の訓練基線としても使用可能 | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-TI2V-5B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-5B テキスト/画像から動画生成重み</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-T2V-A14B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-14B テキストから動画生成重み</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-I2V-A14B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-14B 画像から動画生成重み</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-S2V-14B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.2-S2V-14B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-14B 音声から動画生成重み、話者駆動デジタルヒューマン</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.2-Animate-14B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.2-Animate-14B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.2-14B キャラクター置換・モーション転移重み。リポジトリに複数精度ファイルを含む</td></tr></table> |
| Wan2.1-Fun V1.1 | ビデオ | 本プロジェクトがWan2.1で訓練したV1.1系列。マルチ解像度（512/768/1024）、81フレーム16fps、テキスト/画像から動画、首尾画像、制御生成、カメラ制御をカバー | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-V1.1-1.3B-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-V1.1-1.3Bのテキスト・画像から動画生成の重み。マルチ解像度で訓練され、最初と最後の画像予測をサポートします。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-V1.1-14B-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-14B-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-V1.1-14Bのテキスト・画像から動画生成の重み。マルチ解像度で訓練され、最初と最後の画像予測をサポートします。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-V1.1-1.3B-Control</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-V1.1-1.3Bのビデオ制御重み。Canny、Depth、Pose、MLSDなどの異なる制御条件に対応し、参照画像＋制御条件を使用した制御や軌跡制御をサポートします。512、768、1024のマルチ解像度での動画予測をサポートし、81フレーム、毎秒16フレームで訓練されています。多言語予測に対応しています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-V1.1-14B-Control</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-14B-Control">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-V1.1-14Bのビデオ制御重み。Canny、Depth、Pose、MLSDなどの異なる制御条件に対応し、参照画像＋制御条件を使用した制御や軌跡制御をサポートします。512、768、1024のマルチ解像度での動画予測をサポートし、81フレーム、毎秒16フレームで訓練されています。多言語予測に対応しています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-V1.1-1.3B-Control-Camera</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-V1.1-1.3Bのカメラレンズ制御重み。512、768、1024のマルチ解像度での動画予測をサポートし、81フレーム、毎秒16フレームで訓練されています。多言語予測に対応しています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-V1.1-14B-Control-Camera</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-14B-Control-Camera">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-V1.1-14Bのカメラレンズ制御重み。512、768、1024のマルチ解像度での動画予測をサポートし、81フレーム、毎秒16フレームで訓練されています。多言語予測に対応しています。</td></tr></table> |
| Wan2.1-Fun V1.0 | ビデオ | 本プロジェクトがWan2.1で訓練したV1.0系列。V1.1と同じ能力だがカメラ制御は非対応 | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-1.3B-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-1.3Bのテキスト・画像から動画生成する重み。マルチ解像度で学習され、開始・終了画像予測をサポート。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-14B-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-14Bのテキスト・画像から動画生成する重み。マルチ解像度で学習され、開始・終了画像予測をサポート。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-1.3B-Control</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-Control">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-1.3Bのビデオ制御ウェイト。Canny、Depth、Pose、MLSDなどの異なる制御条件をサポートし、トラジェクトリ制御も利用可能。512、768、1024のマルチ解像度でのビデオ予測をサポートし、81フレーム（1秒間に16フレーム）でトレーニング済みで、多言語予測にも対応しています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-14B-Control</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control">🤖</a></td><td valign="top" style="padding:2px 0;">Wan2.1-Fun-14Bのビデオ制御ウェイト。Canny、Depth、Pose、MLSDなどの異なる制御条件をサポートし、トラジェクトリ制御も利用可能。512、768、1024のマルチ解像度でのビデオ予測をサポートし、81フレーム（1秒間に16フレーム）でトレーニング済みで、多言語予測にも対応しています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-Fun-Reward-LoRAs</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Wan2.1-Fun-Reward-LoRAs">🤖</a></td><td valign="top" style="padding:2px 0;">報酬逆伝播で訓練された整列LoRA</td></tr></table> |
| Wan2.1 | ビデオ | Wan公式重み。テキスト/画像から動画、音声駆動、制御生成をカバー。Wan2.1-Fun系列の訓練基線としても使用可能 | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-T2V-1.3B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B">🤖</a></td><td valign="top" style="padding:2px 0;">1.3B文生视频</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-T2V-14B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B">🤖</a></td><td valign="top" style="padding:2px 0;">14B文生视频</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-I2V-14B-480P</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P">🤖</a></td><td valign="top" style="padding:2px 0;">480P图生视频，是InfiniteTalk的基础模型</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-I2V-14B-720P</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P">🤖</a></td><td valign="top" style="padding:2px 0;">万象2.1-14B-720P 画像→動画モデルの重み</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-VACE-1.3B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B">🤖</a></td><td valign="top" style="padding:2px 0;">1.3B VACE制御と主題参照</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Wan2.1-VACE-14B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Wan-AI/Wan2.1-VACE-14B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B">🤖</a></td><td valign="top" style="padding:2px 0;">14B VACE制御と主題参照</td></tr></table> |
| Self-Forcing / Causal-Forcing | ビデオ | 自己回帰蒸留方案。ストリーミング生成とインタラクティブ生成をカバー | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Self-Forcing</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/gdhe17/Self-Forcing">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/AI-ModelScope/Self-Forcing">🤖</a></td><td valign="top" style="padding:2px 0;">自己回帰蒸留重み、Wan2.1-T2Vと組み合わせて流式・インタラクティブ生成に対応</td></tr></table> |
| CogVideoX-Fun V1.5 | ビデオ | 公式CogVideoX-Fun V1.5重み。マルチ解像度（512/768/1024）、85フレーム8fps、画像から動画と報酬整列をカバー | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-V1.5-5b-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-5b-InP">🤖</a></td><td valign="top" style="padding:2px 0;">公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024）でビデオを予測できます。85フレーム、8フレーム/秒でトレーニングされています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-V1.5-Reward-LoRAs</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-Reward-LoRAs">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-Reward-LoRAs">🤖</a></td><td valign="top" style="padding:2px 0;">公式の報酬逆伝播技術モデルで、CogVideoX-Fun-V1.5が生成するビデオを最適化し、人間の嗜好によりよく合うようにする。</td></tr></table> |
| CogVideoX-Fun V1.1 | ビデオ | 公式CogVideoX-Fun V1.1重み。マルチ解像度（512/768/1024/1280）、49フレーム8fps、画像から動画、ポーズ制御、制御生成、報酬整列をカバー | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-V1.1-2b-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-InP">🤖</a></td><td valign="top" style="padding:2px 0;">公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。参照画像にノイズが追加され、V1.0と比較して動きの幅が広がっています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-V1.1-5b-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-InP">🤖</a></td><td valign="top" style="padding:2px 0;">公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。参照画像にノイズが追加され、V1.0と比較して動きの幅が広がっています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-V1.1-2b-Pose</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Pose">🤖</a></td><td valign="top" style="padding:2px 0;">公式のポーズコントロールビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-V1.1-5b-Pose</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Pose">🤖</a></td><td valign="top" style="padding:2px 0;">公式のポーズコントロールビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-V1.1-2b-Control</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Control">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Control">🤖</a></td><td valign="top" style="padding:2px 0;">公式のコントロールビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。Canny、Depth、Pose、MLSDなどのさまざまなコントロール条件をサポートします。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-V1.1-5b-Control</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Control">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Control">🤖</a></td><td valign="top" style="padding:2px 0;">公式のコントロールビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。Canny、Depth、Pose、MLSDなどのさまざまなコントロール条件をサポートします。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-V1.1-Reward-LoRAs</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-Reward-LoRAs">🤖</a></td><td valign="top" style="padding:2px 0;">公式の報酬逆伝播技術モデルで、CogVideoX-Fun-V1.1が生成するビデオを最適化し、人間の嗜好によりよく合うようにする。</td></tr></table> |
| CogVideoX-Fun V1.0 | ビデオ | 旧版重み。49フレーム8fpsで訓練。V1.1/V1.5に置き換え済み | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-2b-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-2b-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-2b-InP">🤖</a></td><td valign="top" style="padding:2px 0;">公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-5b-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-5b-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-5b-InP">🤖</a></td><td valign="top" style="padding:2px 0;">公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。</td></tr></table> |
| HunyuanVideo | ビデオ | 公式diffusers形式重み。本プロジェクトは推論とLoRA訓練を直接サポート | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">HunyuanVideo</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/hunyuanvideo-community/HunyuanVideo">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo">🤖</a></td><td valign="top" style="padding:2px 0;">文生视频</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">HunyuanVideo-I2V</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo-I2V">🤖</a></td><td valign="top" style="padding:2px 0;">图生视频</td></tr></table> |
| MiniMax-H3 | ビデオ | 公式動画生成重みと本プロジェクトが訓練したControlNet | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">MiniMax-H3</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/MiniMax/MiniMax-H3">🤖</a></td><td valign="top" style="padding:2px 0;">MiniMax-H3公式T2V/I2V重み</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">MiniMax-H3-Fun-Controlnet-Union</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/MiniMax-H3-Fun-Controlnet-Union">🤖</a></td><td valign="top" style="padding:2px 0;">本プロジェクトが訓練したControlNet。複数制御条件と軌跡制御をサポート</td></tr></table> |
| LTX-2 | ビデオ+音声 | 公式DiT音声・動画共同生成重み | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">LTX-2</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Lightricks/LTX-2">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Lightricks/LTX-2">🤖</a></td><td valign="top" style="padding:2px 0;">音声・動画共同生成の公式重み。リポジトリに複数精度ファイルを含む</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">LTX-2.3-Diffusers</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/dg845/LTX-2.3-Diffusers">🤗</a></td><td valign="top" style="padding:2px 8px;">-</td><td valign="top" style="padding:2px 0;">v2.3はコミュニティ変換のdiffusers形式重みを使用。公式重みはLightricks/LTX-2.3を参照</td></tr></table> |
| LongCat-Video | ビデオ | 公式長尺動画生成重み。LoRA訓練をサポート | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">LongCat-Video</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/meituan-longcat/LongCat-Video">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/meituan-longcat/LongCat-Video">🤖</a></td><td valign="top" style="padding:2px 0;">LongCat-Video公式T2V重み</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">LongCat-Video-Avatar</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/meituan-longcat/LongCat-Video-Avatar">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/meituan-longcat/LongCat-Video-Avatar">🤖</a></td><td valign="top" style="padding:2px 0;">LongCat-Video公式アバター/デジタルヒューマン重み</td></tr></table> |
| FantasyTalking | 音声駆動ビデオ | 音声条件付き増分重み。基盤ビデオ重みと音声エンコーダが必要 | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">FantasyTalking</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/acvlab/FantasyTalking">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/amap_cvlab/FantasyTalking">🤖</a></td><td valign="top" style="padding:2px 0;">需搭配Wan2.1-I2V-14B-720P使用</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">wav2vec2-base-960h</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/facebook/wav2vec2-base-960h">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/AI-ModelScope/wav2vec2-base-960h">🤖</a></td><td valign="top" style="padding:2px 0;">音频编码器，放入基础权重目录并命名为audio_encoder</td></tr></table> |
| InfiniteTalk | 音声駆動ビデオ | 音声条件付き増分重み。基盤ビデオ重みと音声エンコーダが必要 | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">InfiniteTalk</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/MeiGen-AI/InfiniteTalk">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/MeiGen-AI/InfiniteTalk">🤖</a></td><td valign="top" style="padding:2px 0;">InfiniteTalk公式オーディオ駆動重み</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">chinese-wav2vec2-base</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/TencentGameMate/chinese-wav2vec2-base">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/TencentGameMate/chinese-wav2vec2-base">🤖</a></td><td valign="top" style="padding:2px 0;">中国語音声エンコーダ</td></tr></table> |
| FlashHead | 音声駆動ビデオ | 公式高品質頭部動作デジタルヒューマン重み | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">SoulX-FlashHead-1_3B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Soul-AILab/SoulX-FlashHead-1_3B">🤖</a></td><td valign="top" style="padding:2px 0;">SoulX FlashHead 1.3B 音声駆動頭部重み。wav2vec音声エンコーダが必要</td></tr></table> |
| MOVA | ビデオ+音声 | 公式MOVA重み | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">MOVA-360p</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/OpenMOSS-Team/MOVA-360p">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/OpenMOSS/MOVA-360p">🤖</a></td><td valign="top" style="padding:2px 0;">画像から動画と音声・動画共同生成</td></tr></table> |
| LingBot | ビデオ | カメラ制御可能なワールドモデル。ディレクトリ構造はWan2.2-I2V-A14Bと一致 | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">lingbot-world-base-cam</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Robbyant/lingbot-world-base-cam">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Robbyant/lingbot-world-base-cam">🤖</a></td><td valign="top" style="padding:2px 0;">カメラ制御基線重み</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">lingbot-video-rewriter-lora</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Robbyant/lingbot-video-rewriter-lora">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Robbyant/lingbot-video-rewriter-lora">🤖</a></td><td valign="top" style="padding:2px 0;">rewriter LoRA。Qwen3.6-27Bで構造化キャプションを生成して使用</td></tr></table> |
| Phantom | ビデオ | 複数主体参照による動画生成の増分重み。Wan2.1-T2Vベース | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Phantom-Wan-1.3B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/bytedance-research/Phantom">🤗</a></td><td valign="top" style="padding:2px 8px;">-</td><td valign="top" style="padding:2px 0;">1.3B版。公式は.pth形式で公開。Personalized_Modelに配置しpredictファイルのtransformer_pathで指定</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Phantom-Wan-14B</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/bytedance-research/Phantom">🤗</a></td><td valign="top" style="padding:2px 8px;">-</td><td valign="top" style="padding:2px 0;">14B版。公式は分割safetensors形式で公開</td></tr></table> |
| Qwen-Image | 画像 | 公式テキストから画像生成・画像編集重み。基線とLoRA訓練をサポート | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Qwen-Image</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Qwen/Qwen-Image">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Qwen/Qwen-Image">🤖</a></td><td valign="top" style="padding:2px 0;">文生图基础权重</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Qwen-Image-2512</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Qwen/Qwen-Image-2512">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Qwen/Qwen-Image-2512">🤖</a></td><td valign="top" style="padding:2px 0;">テキストから画像生成の更新版</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Qwen-Image-Edit</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Qwen/Qwen-Image-Edit">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">🤖</a></td><td valign="top" style="padding:2px 0;">图像编辑</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Qwen-Image-Edit-2509</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Qwen/Qwen-Image-Edit-2509">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2509">🤖</a></td><td valign="top" style="padding:2px 0;">图像编辑更新版本</td></tr></table> |
| Qwen-Image ControlNet | 画像 | 画像制御生成。Canny、Depth、Pose、MLSD、Scribbleをサポート | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Qwen-Image-2512-Fun-Controlnet-Union</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Qwen-Image-2512-Fun-Controlnet-Union">🤖</a></td><td valign="top" style="padding:2px 0;">Qwen-Image-2512のControlNet重み。Canny、Depth、Pose、MLSD、Scribbleなど、複数の制御条件をサポートします。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Qwen-Image-ControlNet-Union</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/InstantX/Qwen-Image-ControlNet-Union">🤖</a></td><td valign="top" style="padding:2px 0;">InstantX提供の同種ControlNet</td></tr></table> |
| Z-Image | 画像 | 公式テキストから画像生成重み | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Z-Image</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Tongyi-MAI/Z-Image">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/Tongyi-MAI/Z-Image">🤖</a></td><td valign="top" style="padding:2px 0;">基础版</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Z-Image-Turbo</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo">🤖</a></td><td valign="top" style="padding:2px 0;">加速版</td></tr></table> |
| Z-Image-Fun | 画像 | 本プロジェクトがZ-Imageで訓練したControlNetと蒸留LoRA。Canny、Depth、Pose、MLSD、Scribble、Grayをサポート | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Z-Image-Fun-Controlnet-Union-2.1</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Z-Image-Fun-Controlnet-Union-2.1">🤖</a></td><td valign="top" style="padding:2px 0;">Z-ImageのControlNet重み、Canny、Depth、Pose、MLSD、ScribbleおよびGrayなど複数の制御条件に対応。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Z-Image-Turbo-Fun-Controlnet-Union</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union">🤖</a></td><td valign="top" style="padding:2px 0;">Z-Image-Turbo用のControlNet重み。Canny、Depth、Pose、MLSDなど複数の制御条件をサポート。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Z-Image-Turbo-Fun-Controlnet-Union-2.1</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1">🤖</a></td><td valign="top" style="padding:2px 0;">Z-Image-TurboのControlNet重み。第1版と比較して、より多くの層に追加され、より長時間トレーニングされています。Canny、Depth、Pose、MLSDなど、複数の制御条件をサポートしています。</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Z-Image-Fun-Lora-Distill</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/Z-Image-Fun-Lora-Distill">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/Z-Image-Fun-Lora-Distill">🤖</a></td><td valign="top" style="padding:2px 0;">これはZ-Image用の蒸留LoRAで、ステップ数とCFGの両方を蒸留します。このモデルはCFGを必要とせず、推論には8ステップを使用します。</td></tr></table> |
| Flux | 画像 | 公式FLUX.1/FLUX.2重みと本プロジェクトが訓練したControlNet | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">FLUX.1-dev</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev">🤖</a></td><td valign="top" style="padding:2px 0;">文生图与图像编辑</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">FLUX.2-dev</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/black-forest-labs/FLUX.2-dev">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev">🤖</a></td><td valign="top" style="padding:2px 0;">第二代官方权重</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">FLUX.2-dev-Fun-Controlnet-Union</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/FLUX.2-dev-Fun-Controlnet-Union">🤖</a></td><td valign="top" style="padding:2px 0;">FLUX.2-dev用ControlNet重み</td></tr></table> |
| ERNIE-Image | 画像 | Baidu公式テキストから画像生成重み | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">ERNIE-Image</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/baidu/ERNIE-Image">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PaddlePaddle/ERNIE-Image">🤖</a></td><td valign="top" style="padding:2px 0;">ERNIE-Image公式画像生成重み</td></tr></table> |
| Lens | 画像 | Microsoft公式カメラ制御重み | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">Lens</td><td valign="top" style="padding:2px 8px;">-</td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/microsoft/Lens">🤖</a></td><td valign="top" style="padding:2px 0;">Lens公式カメラ制御重み</td></tr></table> |
| 補助モデル | - | 生成モデルではなく、報酬整列やデータアノテーションに使用 | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">HPSv3</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/MizzenAI/HPSv3">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/MizzenAI/HPSv3">🤖</a></td><td valign="top" style="padding:2px 0;">報酬逆伝播で使用されるスコアリングモデル</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">Qwen2-VL-7B-Instruct</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct">🤖</a></td><td valign="top" style="padding:2px 0;">動画キャプション生成パイプラインで使用されるマルチモーダルエンコーダ</td></tr></table> |

> 補足説明：
> - 音声駆動・参照系モデル（FantasyTalking、InfiniteTalk、Phantom）は増分重みであり、対応する基盤ビデオ重みと音声エンコーダを同時にダウンロードする必要があります。
> - TurboDiffusion等の蒸留方案は公開重みがなく、`scripts/{model_name}/README_TRAIN*.md`で訓練後、`transformer_path`に指定して使用できます。
> - 重み名は`models/Diffusion_Transformer/`下のフォルダ名と一対一で対応します。同じ系列内の各重みは互換性がないため、推論タスクに応じて選択してください。ここに掲載されていない重みは、本プロジェクトの訓練成果物、または上流の公式リポジトリから取得する必要があります。

# 四、ビデオ作品

### Wan2.1-Fun-V1.1-14B-InP && Wan2.1-Fun-V1.1-1.3B-InP

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d6a46051-8fe6-4174-be12-95ee52c96298" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/8572c656-8548-4b1f-9ec8-8107c6236cb1" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d3411c95-483d-4e30-bc72-483c2b288918" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b2f5addc-06bd-49d9-b925-973090a32800" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/747b6ab8-9617-4ba2-84a0-b51c0efbd4f8" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ae94dcda-9d5e-4bae-a86f-882c4282a367" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/a4aa1a82-e162-4ab5-8f05-72f79568a191" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/83c005b8-ccbc-44a0-a845-c0472763119c" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

### Wan2.1-Fun-V1.1-14B-Control && Wan2.1-Fun-V1.1-1.3B-Control

Generic Control Video + Reference Image:
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          Reference Image
      </td>
      <td>
          Control Video
      </td>
      <td>
          Wan2.1-Fun-V1.1-14B-Control
      </td>
      <td>
          Wan2.1-Fun-V1.1-1.3B-Control
      </td>
  <tr>
      <td>
          <image src="https://github.com/user-attachments/assets/221f2879-3b1b-4fbd-84f9-c3e0b0b3533e" width="100%" controls preload loop></image>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/f361af34-b3b3-4be4-9d03-cd478cb3dfc5" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/85e2f00b-6ef0-4922-90ab-4364afb2c93d" width="100%" controls preload loop></video>
     </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1f3fe763-2754-4215-bc9a-ae804950d4b3" width="100%" controls preload loop></video>
     </td>
  <tr>
</table>


Generic Control Video (Canny, Pose, Depth, etc.) and Trajectory Control:
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/f35602c4-9f0a-4105-9762-1e3a88abbac6" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/8b0f0e87-f1be-4915-bb35-2d53c852333e" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/972012c1-772b-427a-bce6-ba8b39edcfad" width="100%" controls preload loop></video>
     </td>
  <tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ce62d0bd-82c0-4d7b-9c49-7e0e4b605745" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/89dfbffb-c4a6-4821-bcef-8b1489a3ca00" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/72a43e33-854f-4349-861b-c959510d1a84" width="100%" controls preload loop></video>
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/bb0ce13d-dee0-4049-9eec-c92f3ebc1358" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7840c333-7bec-4582-ba63-20a39e1139c4" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/85147d30-ae09-4f36-a077-2167f7a578c0" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

### Wan2.1-Fun-V1.1-14B-Control-Camera && Wan2.1-Fun-V1.1-1.3B-Control-Camera

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          Pan Up
      </td>
      <td>
          Pan Left
      </td>
       <td>
          Pan Right
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/869fe2ef-502a-484e-8656-fe9e626b9f63" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/2d4185c8-d6ec-4831-83b4-b1dbfc3616fa" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/7dfb7cad-ed24-4acc-9377-832445a07ec7" width="100%" controls preload loop></video>
     </td>
  <tr>
      <td>
          Pan Down
      </td>
      <td>
          Pan Up + Pan Left
      </td>
       <td>
          Pan Up + Pan Right
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3ea3a08d-f2df-43a2-976e-bf2659345373" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/4a85b028-4120-4293-886b-b8afe2d01713" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/ad0d58c1-13ef-450c-b658-4fed7ff5ed36" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-5B

解像度-1024

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/34e7ec8f-293e-4655-bb14-5e1ee476f788" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7809c64f-eb8c-48a9-8bdc-ca9261fd5434" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/8e76aaa4-c602-44ac-bcb4-8b24b72c386c" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/19dba894-7c35-4f25-b15c-384167ab3b03" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>


解像度-768

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/0bc339b9-455b-44fd-8917-80272d702737" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/70a043b9-6721-4bd9-be47-78b7ec5c27e9" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d5dd6c09-14f3-40f8-8b6d-91e26519b8ac" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9327e8bc-4f17-46b0-b50d-38c250a9483a" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

解像度-512

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ef407030-8062-454d-aba3-131c21e6b58c" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7610f49e-38b6-4214-aa48-723ae4d1b07e" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1fff0567-1e15-415c-941e-53ee8ae2c841" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/bcec48da-b91b-43a0-9d50-cf026e00fa4f" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-5B-Control

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/53002ce2-dd18-4d4f-8135-b6f68364cabd" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/a1a07cf8-d86d-4cd2-831f-18a6c1ceee1d" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/3224804f-342d-4947-918d-d9fec8e3d273" width="100%" controls preload loop></video>
     </td>
  <tr>
      <td>
          美しい澄んだ目と金髪の若い女性が白い服を着て体をひねり、カメラは彼女の顔に焦点を合わせています。高品質、傑作、最高品質、高解像度、超微細、夢のような。
      </td>
      <td>
          美しい澄んだ目と金髪の若い女性が白い服を着て体をひねり、カメラは彼女の顔に焦点を合わせています。高品質、傑作、最高品質、高解像度、超微細、夢のような。
      </td>
       <td>
          若いクマ。
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ea908454-684b-4d60-b562-3db229a250a9" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ffb7c6fc-8b69-453b-8aad-70dfae3899b9" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d3f757a3-3551-4dcb-9372-7a61469813f5" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

# 五、参考文献
- CogVideo: https://github.com/THUDM/CogVideo/
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate
- Wan2.1: https://github.com/Wan-Video/Wan2.1/
- Wan2.2: https://github.com/Wan-Video/Wan2.2/
- Diffusers: https://github.com/huggingface/diffusers
- Qwen-Image: https://github.com/QwenLM/Qwen-Image
- Self-Forcing: https://github.com/guandeh17/Self-Forcing
- Flux: https://github.com/black-forest-labs/flux
- Flux2: https://github.com/black-forest-labs/flux2
- HunyuanVideo: https://github.com/Tencent-Hunyuan/HunyuanVideo
- ComfyUI-KJNodes: https://github.com/kijai/ComfyUI-KJNodes
- ComfyUI-EasyAnimateWrapper: https://github.com/kijai/ComfyUI-EasyAnimateWrapper
- ComfyUI-CameraCtrl-Wrapper: https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper
- CameraCtrl: https://github.com/hehao13/CameraCtrl

# 六、引用

研究やプロジェクトでVideoX-Funを使用する場合は、以下の形式で引用してください：

```bibtex
@misc{aigc_apps_VideoX_Fun_2026,
  author = {aigc-apps},
  title = {VideoX-Fun: A Video Generation Pipeline for Diffusion Transformer},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/aigc-apps/VideoX-Fun}
}
```

# 七、制限とリスク

- 生成された動画には、特に複雑なシーンでアーティファクトや品質の問題がある場合があります。
- モデルは、細かい詳細、テキストのレンダリング、または特定の芸術スタイルで苦労する場合があります。
- パフォーマンスは、入力プロンプトの品質、解像度、その他のパラメータによって異なります。
- この技術は、誤解を招くコンテンツ（例：ディープフェイク）を作成するために悪用される可能性があります。ユーザーは倫理的な使用に責任を持ちます。
- モデルは、トレーニングデータに存在するバイアスを反映する可能性があります。
- ユーザーは、実在の人物の画像や動画を使用する際、プライバシーと著作権を尊重する必要があります。

責任ある使用を推奨し、本番環境でのセーフガードの実装をお勧めします。

# 八、ライセンス
このプロジェクトは[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)の下でライセンスされています。

CogVideoX-2Bモデル（対応するTransformersモジュール、VAEモジュールを含む）は、[Apache 2.0ライセンス](LICENSE)の下でリリースされています。

CogVideoX-5Bモデル（Transformersモジュール）は、[CogVideoXライセンス](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE)の下でリリースされています。

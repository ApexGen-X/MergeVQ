<div align="center">

<h2><a href="https://arxiv.org/abs/2504.00999">MergeVQ: A Unified Framework for Visual Generation and Representation with Token Merging and Quantization (CVPR 2025)</a></h2>

[Siyuan Li](https://lupin1998.github.io)<sup>1,3*</sup>, [Luyuan Zhang](https://openreview.net/profile?id=~Luyuan_Zhang1)<sup>2*</sup>, [Zedong Wang](https://jacky1128.github.io)<sup>4</sup>, [Juanxi Tian](https://tianshijing.github.io)<sup>3</sup>, [Cheng Tan](https://chengtan9907.github.io)<sup>1,3</sup>, [Zicheng Liu](https://pone7.github.io)<sup>1,3</sup>, [Chang Yu](https://openreview.net/profile?id=~Chang_Yu1)<sup>3</sup>, [Qingsong Xie](https://openreview.net/profile?id=~Qingsong_Xie1)<sup>5†</sup>, [Haoqian Wang](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm)<sup>2</sup>, [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/)<sup>6,7,8†</sup>

<sup>1</sup> Zhejiang University &emsp; <sup>2</sup> Tsinghua University &emsp; <sup>3</sup> Westlake University &emsp; <sup>4</sup> HKUST &emsp; <sup>5</sup> OPPO AI Center &emsp;
<sup>6</sup> CAIR, HKISI-CAS &emsp; <sup>7</sup> MAIS CASIA &emsp; <sup>8</sup> University of Chinese Academy of Sciences

<sup>*</sup> Equal Contributions;  <sup>†</sup> Corresponding Authors.

<!-- IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025 -->

</div>

<p align="center">
<a href="https://arxiv.org/abs/2504.00999" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2504.00999-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/ApexGen-X/MergeVQ/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" /></a>
<!-- <a href="https://colab.research.google.com/github/Westlake-AI/MogaNet/blob/main/demo.ipynb" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a> -->
<!-- <a href="https://huggingface.co/MogaNet" alt="Huggingface">
    <img src="https://img.shields.io/badge/huggingface-MogaNet-blueviolet" /></a> -->
</p>

![mergevq_framework](https://github.com/user-attachments/assets/a3e22ba0-6f0d-43bb-bf38-cf628ec1aa41)

Masked Image Modeling (MIM) with Vector Quantization (VQ) has achieved great success in both self-supervised pre-training and image generation. However, most existing methods struggle to address the trade-off in shared latent space for generation quality vs. representation learning and efficiency. To push the limits of this paradigm, we propose MergeVQ, which incorporates token merging techniques into VQ-based autoregressive generative models to bridge the gap between visual generation and representation learning in a unified architecture. During pre-training, MergeVQ decouples top-k semantics from latent space with a token merge module after self-attention blocks in the encoder for subsequent Look-up Free Quantization (LFQ) and global alignment and recovers their fine-grained details through cross-attention in the decoder for reconstruction. As for the second-stage generation, we introduce MergeAR, which performs KV Cache compression for efficient raster-order prediction. Experiments on ImageNet verify that MergeVQ as an AR generative model achieves competitive performance in both representation learning and image generation tasks while maintaining favorable token efficiency and inference speed.

🤗 HuggingFace Daily Papers Top-1: [https://huggingface.co/papers/2504.00999](https://huggingface.co/papers/2504.00999) 

## Catalog

We plan to release implementations of MergeVQ in a few months (before CVPR2025 taking place). Please watch us for the latest release and welcome to open issues for discussion! Currently, we have released the basic implementations of MergeVQ tokenizers.

## 📖 Implementations

### 🛠️ Installation

#### GPU
- **Environments**: We have tested on `Python3.10.0` + `torch2.1.0+cuda12.1`, and `Python 3.8.8` + `torch==1.3.0+cuda11.8`, and other versions may also work.
- **Dependencies**: `pip install -r requirements.txt`
Here is an example of installing with `torch2.1.0+cuda12.1` from scratch:
```sh
conda create -n mergevq python=3.10.0
conda activate mergevq
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

#### NPU
- **Env**: `Python 3.9.16` and [`CANN 8.0.T13`](https://www.hiascend.com/en/software/cann)
- **Main Dependencies**: `torch=2.1.0+cpu` + `torch-npu=2.1.0.post3-20240523` + [`Lightning`](https://github.com/hipudding/pytorch-lightning/tree/npu_support)
- **Other Dependencies**: see in `requirements.txt`

#### Datasets Preparation
We use ILSVRC2012 ImageNet with [training set](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) and [validation set](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) at the root, which could be downloaded as untared as follows:
```
.cache/imagenet
└── train/
    ├── n01440764
        ├── n01440764_10026.JPEG
        ├── n01440764_10027.JPEG
        ├── ...
    ├── n01443537
    ├── ...
└── val/
    ├── n01440764
    ├── n01443537
    ├── ...
```
When start training or evaluation, these files will be generated under `.cache/imagenet/train` and `.cache/imagenet/val`, including `filelist.txt`, `imagenet_idx_to_synset.yaml`, `synset_human.txt`, and `validation_synset.txt`. If you want to use a custom dataset or ImageNet at the other file path, please specify `cachedir` for `taming.data.imagenet.ImageNetTrain` in the training config file.

#### Pre-training Models
If you are not available to access `https://huggingface.co/` smoothly, we have two solutions.
* Export to the mirror website (`https://hf-mirror.com`) and start training directly:
```sh
export HF_ENDPOINT=https://hf-mirror.com
```
Manually download the following pre-trained models from the offical or mirror websites and copy them to the cache folder as follows, or modify the config file with the path of local huggingface models.
```
/root/.cache/huggingface/hub
└── models--facebook--dinov2-base
└── models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K
└── models--timm--vit_base_patch14_dinov2.lvd142m
```
```python
from timm import create_model
teacher_weights = create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True).state_dict()
teacher_weights = create_model("vit_base_patch16_clip_224.laion2b", pretrained=True).state_dict()
from transformers import AutoModel
dist_model = AutoModel.from_pretrained("facebook/dinov2-base")
```

### Stage I: Training of Visual Tokenizer

#### 🚀 Training Scripts
* $256\times 256$ MergeVQ-d64 (G+R) Tokenizer Training with multiple nodes:
```sh
bash scripts/train_tokenizer/run_256_GR_d64_multi.sh MASTER_ADDR MASTER_PORT NODE_RANK
```
Or you can start training and evaluation on a single node, taking 8xA100-80G with a batch size of 16 and 2 times gradient accumulations as an example:
```sh
bash scripts/train_tokenizer/run_256_GR_d64_single.sh
```

* $256\times 256$ MergeVQ-d96 (G+R) Tokenizer Training with multiple nodes:
```sh
bash scripts/train_tokenizer/run_256_GR_d96_multi.sh MASTER_ADDR MASTER_PORT NODE_RANK
```
Or you can start training and evaluation on a single node, taking 8xA100-80G with a batch size of 16 and 2 times gradient accumulations as an example:
```sh
bash scripts/train_tokenizer/run_256_GR_d96_single.sh
```

* $256\times 256$ MergeVQ-d64 (G) Tokenizer Training with multiple nodes:
```sh
bash scripts/train_tokenizer/run_256_G_d64_multi.sh MASTER_ADDR MASTER_PORT NODE_RANK
```
Or you can start training and evaluation on a single node, taking 8xA100-80G with a batch size of 8 and 4 times gradient accumulations as an example:
```sh
bash scripts/train_tokenizer/run_256_G_d64_single.sh
```

#### Evaluation Scripts
We gather evaluation scripts of experiments above into one bash file, which can be executed with modified path to config files, results, and checkpoints:
```sh
bash scripts/evaluation/evaluation_mergevq.sh
```

#### Note of Errors
If the some errors occur during training, you may solve them with the following steps:
* The version of `timm`. The low version of `timm` like `0.6.13` will cause errors in building Transformer Blocks, which can be solved by `pip install timm==0.9.11`.
* Errors in building up ImageNet dataset. Although the meta files of ImageNet will be generated automatically, you might copy our preprocess meta files manually if it cannot be generated.
<!-- * The assertion error of `accumulate_grad_batches` from `lightning`. Since we manually use `accumulate_grad_batches` in config files to setup gradient accumulation, please replace the source file `configuration_validator.py` with our modified version in lightning.
```sh
cp -r scripts/.modify_lightning/configuration_validator.py /root/anaconda3/envs/maskgit/lib/python3.10/site-packages/lightning/pytorch/trainer/configuration_validator
``` -->

<!-- 
#### 🚀 Evaluation Scripts
* $128\times 128$ Tokenizer Evaluation
```
bash scripts/evaluation/evaluation_128.sh
```

* $256\times 256$ Tokenizer Evaluation
```
bash scripts/evaluation/evaluation_256.sh
``` -->

#### 🍺 Performance and Models (Updating)

**Tokenizer**
| Method | Type | #Tokens | Train Size | Epoch | Codebook Size | rFID (Full) | rFID (Merge) | Checkpoint |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Open-MAGVIT2 | 2D | $16^2$ | $256^2$ | 270 | 2^18 | 1.53 (256) | - | [ckpt](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_256_L.ckpt) |
| MergeVQ-d32 (G) | 1D | [256, 1024] | $256^2$ | 200 | 2^18 | 0.48 (1024) | 0.80 (256) | TODO |
| MergeVQ-d64 (G) | 1D | [256, 1024] | $256^2$ | 100 | 2^18 | 0.49 (1024) | 0.91 (256) | TODO |
| MergeVQ-d64 (G) | 1D | [256, 1024] | $256^2$ | 200 | 2^18 | 0.43 (1024) | 0.83 (256) | TODO |
| MergeVQ-d32 (G+R) | 1D | [144, 256] | $256^2$ | 270 | 2^18 | 1.27 (256) | 1.74 (144) | TODO |
| MergeVQ-d64 (G+R) | 1D | [144, 256] | $256^2$ | 270 | 2^18 | 1.12 (256) | 1.48 (144) | TODO |
| MergeVQ-d96 (G+R) | 1D | [144, 256] | $256^2$ | 200 | 2^18 | 1.03 (256) | 1.33 (144) | TODO |

### Stage II: Training of Auto-Regressive Models

#### 🚀 Training Scripts
Please see in scripts/train_autogressive/run.sh for different model configurations.
```
bash scripts/train_autogressive/run.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

#### 🚀 Sample Scripts
Please see in scripts/train_autogressive/run.sh for different sampling hyper-parameters for different scale of models.
```
bash scripts/evaluation/sample_npu.sh or scripts/evaluation/sample_gpu.sh Your_Total_Rank
```

<!-- #### 🍺 Performance and Models
| Method | Params| #Tokens | FID | IS | Checkpoint |
|:------:|:-----:|:-------:|:---:|:--:|:----------:|
|Open-MAGVIT2| 343M | 16 $\times$ 16 | 3.08 | 258.26 | [AR_256_B](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/AR_256_B.ckpt)|
|Open-MAGVIT2| 804M | 16 $\times$ 16 | 2.51 | 271.70 | [AR_256_L](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/AR_256_L.ckpt)|
|Open-MAGVIT2| 1.5B | 16 $\times$ 16 | 2.33 | 271.77 | [AR_256_XL](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/AR_256_XL.ckpt)| -->


## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- [VQGAN](https://github.com/CompVis/taming-transformers): Taming Transformers for High-Resolution Image Synthesis.
- [ToMe](https://github.com/facebookresearch/ToMe): Token Merging: Your ViT but Faster.
- [LlamaGen](https://github.com/FoundationVision/LlamaGen): Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation.
- [SEED-Voken (OpenMAGVIT2)](https://github.com/TencentARC/SEED-Voken): SEED-Voken: A Series of Powerful Visual Tokenizers.
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models): PyTorch image models, scripts, pretrained weights.

## Citation

If you find this repository helpful, please consider citing:
```
@inproceedings{cvpr2025mergevq,
    title={MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization},
    author={Li, Siyuan and Zhang, Luyuan and Wang, Zedong and Tian, Juanxi and Tan, Cheng and Liu, Zicheng and Yu, Chang and Xie, Qingsong and Lu, Haonan and Wang, Haoqian and Lei, Zhen},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

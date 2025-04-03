<div align="center">

<h2><a href="https://arxiv.org/abs/2504.00999">MergeVQ: A Unified Framework for Visual Generation and Representation with Token Merging and Quantization [CVPR 2025]</a></h2>

<!-- ### [arXiv Paper](https://arxiv.org/abs/2504.00999) -->

[Siyuan Li](https://lupin1998.github.io)<sup>1,3*</sup>, [Luyuan Zhang](https://openreview.net/profile?id=~Luyuan_Zhang1)<sup>2*</sup>, [Zedong Wang](https://jacky1128.github.io)<sup>4</sup>, [Juanxi Tian](https://tianshijing.github.io)<sup>3</sup>, [Cheng Tan](https://chengtan9907.github.io)<sup>1,3</sup>, [Zicheng Liu](https://pone7.github.io)<sup>1,3</sup>, [Chang Yu](https://openreview.net/profile?id=~Chang_Yu1)<sup>3</sup>, [Qingsong Xie](https://openreview.net/profile?id=~Qingsong_Xie1)<sup>5†</sup>, [Haoqian Wang](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm)<sup>2</sup>, [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/)<sup>6,7,8†</sup>

<sup>1</sup> Zhejiang University &emsp; <sup>2</sup> Tsinghua University &emsp; <sup>3</sup> Westlake University &emsp; <sup>4</sup> HKUST &emsp; <sup>5</sup> OPPO AI Center &emsp; <sup>6</sup> CAIR, HKISI-CAS &emsp; <sup>7</sup> MAIS CASIA &emsp; <sup>8</sup> University of Chinese Academy of Sciences

<sup>*</sup> Equal Contributions. <sup>†</sup> Corresponding Authors.

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

Masked Image Modeling (MIM) with Vector Quantization (VQ) has achieved great success in both self-supervised pre-training and image generation. However, most existing methods struggle to address the trade-off in shared latent space for generation quality vs. representation learning and efficiency.
To push the limits of this paradigm, we propose MergeVQ, which incorporates token merging techniques into VQ-based autoregressive generative models to bridge the gap between visual generation and representation learning in a unified architecture. During pre-training, MergeVQ decouples top-k semantics from latent space with a token merge module after self-attention blocks in the encoder for subsequent Look-up Free Quantization (LFQ) and global alignment and recovers their fine-grained details through cross-attention in the decoder for reconstruction. As for the second-stage generation, we introduce MergeAR, which performs KV Cache compression for efficient raster-order prediction.
Experiments on ImageNet verify that MergeVQ as an AR generative model achieves competitive performance in both representation learning and image generation tasks while maintaining favorable token efficiency and inference speed.

HuggingFace: [https://huggingface.co/papers](https://huggingface.co/papers/2504.00999) (#1 Paper of the day⬆️)
## Catalog

We plan to release implementations of MergeVQ in a few months (before CVPR2025 taking place). Please watch us for the latest release and welcome to open issues for discussion!

## Citation

<!-- If you find this repository helpful, please consider citing: -->
```
@inproceedings{cvpr2025mergevq,
    title={MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization},
    author={Li, Siyuan and Zhang, Luyuan and Wang, Zedong and Tian, Juanxi and Tan, Cheng and Liu, Zicheng and Yu, Chang and Xie, Qingsong and Lu, Haonan and Wang, Haoqian and Lei, Zhen},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}

@misc{li2025mergevqunifiedframeworkvisual,
      title={MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization}, 
      author={Siyuan Li and Luyuan Zhang and Zedong Wang and Juanxi Tian and Cheng Tan and Zicheng Liu and Chang Yu and Qingsong Xie and Haonan Lu and Haoqian Wang and Zhen Lei},
      year={2025},
      eprint={2504.00999},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.00999}, 
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

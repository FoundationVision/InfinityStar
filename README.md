# Infinity**⭐️**: Uniﬁed **S**pace**T**ime **A**uto**R**egressive Modeling for Visual Generation


<div align="center">

[![demo platform](https://img.shields.io/badge/Play%20with%20Infinity%21-Infinity%20demo%20platform-lightblue)](http://opensource.bytedance.com/discord/invite)&nbsp;
<!-- [![arXiv](https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages)](https://foundationvision.github.io/infinity.project/)&nbsp; -->
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2511.04675-b31b1b.svg)](https://arxiv.org/abs/2511.04675)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-FoundationVision/Infinity-yellow)](https://huggingface.co/FoundationVision/InfinityStar)&nbsp;
<!-- [![Replicate](https://replicate.com/chenxwh/infinity/badge)](https://replicate.com/chenxwh/infinity)&nbsp; -->

</div>
<p align="center" style="font-size: larger;">
  <a href="http://arxiv.org/abs/2511.04675">Infinity⭐️: Uniﬁed Spacetime AutoRegressive Modeling for Visual Generation</a>
</p>


<!-- <p align="center">
<img src="assets/show_images.jpg" width=95%>
<p> -->

## 🔥 Updates!!
* Nov 7, 2025: 🔥 Paper, Training and Inference Codes && Checkpoints && Demo Websites released!
* Sep 18, 2025: 🎉 InfinityStar is accepted as NeurIPS 2025 Oral.

## 🕹️ Try and Play with Infinity!

We provide a [demo website](http://opensource.bytedance.com/discord/invite) for you to play with InfinityStar and generate videos. Enjoy the fun of bitwise video autoregressive modeling!


## 📑 Open-Source Plan
  - [x] Training Code 
  - [x] Web Demo 
  - [x] InfinityStar Inference Code
  - [x] InfinityStar Models Checkpoints
  - [ ] InfinityStar-Interact Checkpoints & Inference Code



## 📖 Introduction
We introduce InfinityStar, a unified spacetime autoregressive framework for high-resolution image and dynamic video synthesis. Building on the recent success of autoregressive modeling in both vision and language, our purely discrete approach jointly captures spatial and temporal dependencies within a single architecture. This unified design naturally supports a variety of generation tasks such as text-to-image, text-to-video, image-to-video, and long-duration video synthesis via straightforward temporal autoregression. Through extensive experiments, InfinityStar scores 83.74 on VBench, outperforming all autoregressive models by large margins, even surpassing diffusion competitors like HunyuanVideo. Without extra optimizations, our model generates a 5s, 720p video approximately 10x faster than leading diffusion-based methods. To our knowledge, InfinityStar is the first discrete autoregressive video generator capable of producing industrial-level 720p videos. We release all code and models to foster further research in efficient, high-quality video generation.




## Installation
1. We use FlexAttention to speedup training, which requires `torch>=2.5.1`.
2. Install other pip packages via `pip3 install -r requirements.txt`.


## Training Scripts
We provide a comprehensive workflow for training and finetuning our model, covering data organization, feature extraction, and training scripts. For detailed instructions, please refer to `data/README.md`.

## Inference
*   **720p Video Generation:** 
    Use `tools/infer_video_720p.py` to generate 5-second videos at 720p resolution. Due to the high computational cost of training, our released 720p model is trained for 5-second video generation. This script also supports image-to-video generation by specifying an image path.
    ```bash
    python3 tools/infer_video_720p.py
    ```

*   **480p Variable-Length Video Generation:**
    We also provide an intermediate checkpoint for 480p resolution, capable of generating videos of 5 and 10 seconds. Since this model is not specifically optimized for Text-to-Video (T2V), we recommend using the experimental Image-to-Video (I2V) and Video-to-Video (V2V) modes for better results. To specify the video duration, you can edit the `generation_duration` variable in `tools/infer_video_480p.py` to either 5 or 10. This script also supports image-to-video and video continuation by providing a path to an image or a video.
    ```bash
    python3 tools/infer_video_480p.py
    ```


## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:

```
@article{InfinityStar,
  title={InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation},
  author={Jinlai Liu and jian Han and Bin Yan and Hui Wu and Fengda Zhu and Xing Wang and Yi Jiang and Bingyue Peng and Zehuan Yuan},
  journal={Advances in neural information processing systems},
  year={2025},
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

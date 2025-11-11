<p align="center">
  <img src="assets/logo.png" width="400" style="border:none;box-shadow:none;border-radius:0;background:none;">
<p>
  
# Infinity**⭐️**: Uniﬁed **S**pace**T**ime **A**uto**R**egressive Modeling for Visual Generation


<div align="center">

[![demo platform](https://img.shields.io/badge/Play%20with%20Infinity%21-Infinity%20demo%20platform-lightblue)](http://opensource.bytedance.com/discord/invite)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2511.04675-b31b1b.svg)](https://arxiv.org/abs/2511.04675)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-FoundationVision/Infinity-yellow)](https://huggingface.co/FoundationVision/InfinityStar)&nbsp;

</div>
<p align="center" style="font-size: larger;">
  <a href="http://arxiv.org/abs/2511.04675">Infinity⭐️: Uniﬁed Spacetime AutoRegressive Modeling for Visual Generation</a>
</p>

<!-- <p align="center">
<img src="assets/show_images.jpg" width=95%>
<p> -->

---
## 🔥 Updates!!
* Nov 7, 2025: 🔥 Paper, Training and Inference Codes && Checkpoints && Demo Website released!
* Sep 18, 2025: 🎉 InfinityStar is accepted as NeurIPS 2025 Oral.

## 🕹️ Try and Play with Infinity⭐️!

We provide a [demo website](http://opensource.bytedance.com/discord/invite) for you to play with InfinityStar and generate videos. Enjoy the fun of bitwise video autoregressive modeling!

## ✨ Overview
We introduce InfinityStar, a unified spacetime autoregressive framework for high-resolution image and dynamic video synthesis.

- 🧠 **Unified Spacetime Model**: A purely discrete, autoregressive approach that jointly captures spatial and temporal dependencies within a single, elegant architecture.
  
- 🎬 **Versatile Generation**: This unified design naturally supports a variety of generation tasks such as **text-to-image**, **text-to-video**, **image-to-video**, and **long interactive video synthesis** via straightforward temporal autoregression.
  
- 🏆 **Leading Performance & Speed**: Through extensive experiments, InfinityStar scores **83.74** on VBench, outperforming all autoregressive models by large margins, even surpassing diffusion competitors like HunyuanVideo, approximately **10x** faster than leading diffusion-based methods.
  
- 📖 **Pioneering High-Resolution Autoregressive Generation**: To our knowledge, InfinityStar is the first discrete autoregressive video generator capable of producing industrial-level 720p videos, setting a new standard for quality in its class.


### 🔥 Unified modeling for image, video generation and long interactive video synthesis 📈:

<div align="left">
    <img src="assets/framework.png" alt="" style="width: 100%;" />
</div>

## 🎬 Video Demos
#### General Aesthetics
<div align="left">
<video src="https://github.com/user-attachments/assets/14e2b18b-9234-42ce-bdab-670faeef4b2a" width="100%" controls autoplay loop></video>
</div>

####  Anime & 3D Animation
<div align="left">
<video src="https://github.com/user-attachments/assets/478e9571-b550-4c23-a567-6fee9a0afb5b" width="100%" controls autoplay loop></video>
</div>

#### Motion
<div align="left">
<video src="https://github.com/user-attachments/assets/adab669b-d38f-4607-9a52-32d8d0bf0e53" width="100%" controls autoplay loop></video>
</div>

#### Extended Application: Long Interactive Videos
<div align="center">
<video src="https://github.com/user-attachments/assets/411666a6-563d-4551-a3f8-dc5de00436c1" width="100%" controls autoplay loop></video>
</div>

## Benchmark

### Achieve sota performance on image generation benchmark:

<div align="left">
    <img src="assets/Infinitystar_image_gen_benchmark.png" alt="Image Generation Evaluation" style="width: 100%;" />
</div>

### Achieve sota performance on video generation benchmark:

<div align="left">
    <img src="assets/Infinitystar_videogen_benchmark.png" alt="" style="width: 100%;" />
</div>

### Surpassing diffusion competitors like HunyuanVideo*:

<div align="left">
    <img src="assets/Infinitystar_videogen_humaneval.png" alt="" style="width: 100%;" />
</div>


## Visualization

### Text to image examples

<div align="left">
    <img src="assets/supp_show_images.png" alt="Text to Image Examples" style="width: 100%;" />
</div>

### Image to video examples

<div align="left">
    <img src="assets/i2v_examples.png" alt="Image to Video Examples" style="width: 100%;" />
</div>

### Video extrapolation examples

<div align="left">
    <img src="assets/v2v_examples.png" alt="Video Extrapolation Examples" style="width: 100%;" />
</div>

## 📑 Open-Source Plan
  - [x] Training Code 
  - [x] Web Demo 
  - [x] InfinityStar Inference Code
  - [x] InfinityStar Models Checkpoints
  - [x] InfinityStar-Interact Inference Code
  - [ ] InfinityStar-Interact Checkpoints


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

*   **480p Long Interactive Video Generation:**
    Use `tools/infer_interact_480p.py` to generate a long interactive video in 480p. This script supports interactive video generation. You can provide a reference video and multiple prompts. The model will generate a video interactively with your assistance. 
    ```bash
    python3 tools/infer_interact_480p.py
    ```

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:

```
@Article{VAR,
      title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction}, 
      author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
      year={2024},
      eprint={2404.02905},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@misc{Infinity,
    title={Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis}, 
    author={Jian Han and Jinlai Liu and Yi Jiang and Bin Yan and Yuqi Zhang and Zehuan Yuan and Bingyue Peng and Xiaobing Liu},
    year={2024},
    eprint={2412.04431},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.04431}, 
}
```

```
@misc{InfinityStar,
      title={InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation}, 
      author={Jinlai Liu and Jian Han and Bin Yan and Hui Wu and Fengda Zhu and Xing Wang and Yi Jiang and Bingyue Peng and Zehuan Yuan},
      year={2025},
      eprint={2511.04675},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.04675}, 
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

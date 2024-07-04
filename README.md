# A Unified Framework for 3D Scene Understanding

#### [Wei Xu](https://scholar.google.com.hk/citations?hl=zh-CN&user=oMvFn0wAAAAJ)</sup>\*, [Chunsheng Shi](https://github.com/chunshengshi)</sup>\*, [Sifan Tu](https://github.com/SamorsetTuska), [Xin Zhou](https://lmd0311.github.io/), [Dingkang Liang](https://scholar.google.com/citations?user=Tre69v0AAAAJ&hl=zh-CN), [Xiang Bai](https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=zh-CN)
Huazhong University of Science and Technology<br>
(*) equal contribution.

An officical implementation of "A Unified Framework for 3D Scene Understanding".<br>
Code will be coming soon.

[[Project Page](https://dk-liang.github.io/UniSeg3D/)]  [[Arxiv Paper](https://arxiv.org/abs/2407.03263)]

### Abstract
We propose UniSeg3D, a unified 3D segmentation framework that achieves panoptic, semantic, instance, interactive, referring, and open-vocabulary semantic segmentation tasks within a single model. Most previous 3D segmentation approaches are specialized for a specific task, thereby limiting their understanding of 3D scenes to a task-specific perspective. In contrast, the proposed method unifies six tasks into unified representations processed by the same Transformer. It facilitates inter-task knowledge sharing and, therefore, promotes comprehensive 3D scene understanding. To take advantage of multi-task unification, we enhance the performance by leveraging task connections. Specifically, we design a knowledge distillation method and a contrastive learning method to transfer task-specific knowledge across different tasks. Benefiting from extensive inter-task knowledge sharing, our UniSeg3D becomes more powerful. Experiments on three benchmarks, including the ScanNet20, ScanRefer, and ScanNet200, demonstrate that the UniSeg3D consistently outperforms current SOTA methods, even those specialized for individual tasks. We hope UniSeg3D can serve as a solid unified baseline and inspire future work.

### Overview
![Introduction](content/introduction.png)

### Experiment
![Experiment](content/comparison.png)

### Visulizations on six 3D segmentation tasks
![visualization](content/visualization.png)

## Citation
```
@article{UniSeg3D,
      title={A Unified Framework for 3D Scene Understanding}, 
      author={Wei Xu and Chunsheng Shi and Sifan Tu and Xin Zhou and Dingkang Liang and Xiang Bai},
      journal={arXiv preprint arXiv:2407.03263},
      year={2024}
}
```








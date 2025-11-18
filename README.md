## RCNet: Deep Recurrent Collaborative Network for Multi-View Low-Light Image Enhancement

[Paper](https://ieeexplore-ieee-org.ezproxy.cityu.edu.hk/abstract/document/10820442?casa_token=VLc0UgoKKbsAAAAA:wgI8GPtGxkyFxa6vX-zkmDYF_YLvMowHSr7dlmJ7xPdDvju3EsfGRHUvy5XvEiJy-0Xj_S-X) | [arXiv](https://arxiv.org/abs/2409.04363)


[Hao Luo](https://github.com/hluo29/RCNet)<sup>1</sup>, [Baoliang Chen](https://baoliang93.github.io/)<sup>2</sup>, [Lingyu Zhu](https://github.com/hluo29/RCNet)<sup>1</sup>, [Peilin Chen](https://github.com/hluo29/RCNet)<sup>1</sup>, [Shiqi Wang](https://scholar.google.com.hk/citations?hl=en&user=Pr7s2VUAAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup>

<sup>1</sup>City University of Hong Kong<br><sup>2</sup>South China Normal University

<p align="center">
    <img src="docs/static/images/fig-dataset-statistics.png">
</p>

---

<p align="center">
    <img src="docs/static/images/fig3-framework.png">
</p>

:star:If RCNet is helpful for you, please help star this repo. Thanks!:hugs:

## Table Of Contents

- [TODO](#todo)
- [Data Preparation](#data)
- [Installation](#env)
- [Inference](#inference)
- [Training](#training)

## <a name="todo"></a>TODO

- [x] Update link to paper and Release the MVLT dataset.
- [x] Provide a runtime environment installation.
- [x] Release inference code and pretrained models.
- [x] Release training code.

The code and pretrained model have be made available now!

## <a name="data"></a>Data Preparation
Google Driver of MVLT Dataset: https://drive.google.com/drive/folders/1QR5TgocnWFGx2Qk75yEfqmJwvnD2CqWx?usp=sharing

## <a name="env"></a>Installation
This codebase was tested with the following environment configurations. It may work with other versions.

- Ubuntu 20.04
- CUDA 11.1
- Python 3.8
- PyTorch 1.7.0 + cu110
- spatial-correlation-sampler 0.4.0

## <a name="inference"></a>Inference
```shell
python test.py
```

## <a name="training"></a>Training
```shell
python train.py
```

## Citation

Please cite us if our work is useful for your research.

```
@article{luo2025rcnet,
  title={{RCNet}: Deep recurrent collaborative network for multi-view low-light image enhancement},
  author={Luo, Hao and Chen, Baoliang and Zhu, Lingyu and Chen, Peilin and Wang, Shiqi},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  volume={27},
  pages={2001-2014},
  publisher={IEEE}
}
```

## Acknowledgement

This project is based on [MuCAN](https://github.com/dvlab-research/Simple-SR). Thanks for their awesome work.

## Contact

If you have any questions, please feel free to contact with me at `hluo29-c@my.cityu.edu.hk`.

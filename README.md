# bamEdge2

Fine-tuned [LaMa](https://github.com/advimman/lama) for fixing compositing edges in Nuke.

My idea was to build something that is quick enough to use in a normal Nuke compositing workflow. It is not meant to fix all broken edges. It is meant as an additional tool that might help artists to fix a bad edge. As LaMa is using FFC it is better on repetitive structures.

The model was trained specifically on synthetic compositing data (masked subjects over black).

## How it works

**Fair warning:** I put a lot of "manual" work into the data augmentation and fine tuning. BUT I vibecoded the nuke plugin itself.

The node takes a premultiplied RGBA input. The alpha defines where the edge fixing happens:

1. The alpha is binarized and eroded inward by **Edge Size** pixels
2. The ring between the original and eroded alpha is sent to the model
3. The model fills in what a clean edge should look like
4. The result is blended back using a soft transition controlled by **Blend Transition**

Two model variants are embedded in the plugin: **bamEdge2A** and **bamEdge2B**, selectable from a pulldown.

**Util channels**
- bE2_model_mask: the edge band passed to the model
- bE2_transition_mask: the blend weight used for compositing

## Get the plugin

[BamEdge2 Nuke Plugin for Linux Nuke 16.0v7](https://bewegtbildbau.de/bamEdge2/bamEdge2_linux_nuke16_0v7.zip)

I tested this version on a laptop running on **Linux** without any GPU in Nuke 16.

[BamEdge2 Nuke Plugin for Windows Nuke 17.0v3](https://bewegtbildbau.de/bamEdge2/bamEdge2_windows_17_0v2.zip)

I tested this version on a workstation with a RTX 3090 on **Windows** using Nuke 17.

These are the only versions I compiled. If you want anything else you have to build it yourself. :)

## Building

### Dependencies

- Nuke
- Libtorch
- The two model files: [Download Models](https://bewegtbildbau.de/bamEdge2/models.zip)

### Linux

1. Download Libtorch https://pytorch.org/

You need whatever your Nuke version is using:
https://learn.foundry.com/nuke/17.0v1/content/misc/studio_third_party_libraries.html

2. Download the two model files and extract them: [Download Models](https://bewegtbildbau.de/bamEdge2/models.zip)

3. Configure and build
```bash
cd nuke/cpp
mkdir build && cd build
cmake .. \
  -DLIBTORCH_INCLUDE_DIR=/path/to/libtorch \
  -DNUKE_ROOT=$HOME/Nuke16.0v7 \
  -DMODELS_DIR=/path/to/models
cmake --build . -j$(nproc)
cmake --install .
```

4. Add plugin to nuke

### Windows

Same steps, but you need [NASM](https://www.nasm.us/). Adjust the Nuke root:

```bat
cmake .. ^
  -DLIBTORCH_DIR=path\to\libtorch ^
  -DNUKE_ROOT="C:\path\to\Nuke17.0v2" ^
  -DMODELS_DIR=path\to\models
  -NASM_EXECUTABLE=path\to\nasm.exe ^
cmake --build . --config Release
cmake --install .
```

The install step creates a `build/bamEdge2/` folder containing the plugin `.so`/`.dll`.

## Model & training

**Base model:** Big LaMa FFC-ResNet generator with Fourier convolutions, 18 residual blocks, fine-tuned from the official pretrained checkpoint.

**Data preparation:**

Training data was built synthetically:

- **Source images:** CelebA-HQ and FFHQ. Human subjects because hair is the hardest edge case.
- **Segmentation:** SAM2 for mask generation and ViTMatte for improving masks
- **Synthetic comp:** each segment composited over black — the model never sees real production footage
- **Edge masks:** three erosion levels (8 / 16 / 24 px) per segment, giving the model varied edge widths to train on
- **Scale:** trained on over 800.000 segmented images

**Training:** I ran it on RunPod (with one RTX 4090). Data augmentation took longer than the training if I remember correctly. As I did several tests I don't recollect exactly but I think it took around 2 days for both together.

## License

The plugin source code is licensed under [Apache 2.0](LICENSE).

The pre-trained model files (`bam_model_a.pt`, `bam_model_b.pt`) were trained on the
[FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) which is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

I honestly have no clue what that means for the weights. Time will tell. Maybe. If you want to be on the safe side only use it for non-commercial.

## Acknowledgements

- [LaMa](https://github.com/advimman/lama)
- [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
- [ViTMatte](https://github.com/hustvl/ViTMatte)
- [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset)
- [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans)

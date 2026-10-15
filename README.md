# CAM-DGD: Context-Aware Multimodal Decoupled Graph Distillation for Emotion Recognition

CAM-DGD is a multimodal model that predicts emotion and sentiment from text, audio, and video, and also looks at the visual surroundings of the speaker (background and scene), not only the speaker's face and voice.

## Overview

Most multimodal emotion models mix all inputs into one representation. This project tries three ideas instead:

1. Use context: Alongside text, audio, and video, the model gets a fourth stream built from sampled frames: a version of each frame with the speaker masked out, and a depth map of the scene.

2. Separate what is shared from what is specific: For each modality, the model learns one part that is shared across modalities (carrying emotion information) and one part that is private to that modality.

3. Let modalities teach each other: Graph distillation lets modalities pass knowledge to one another, with learned weights deciding who guides whom.

## Architecture

<img width="898" height="449" alt="Architecture_Final" src="https://github.com/user-attachments/assets/c32e2996-8c44-489f-9436-fd75a4a987f5" />

The model has five stages.

1. Feature extraction (one branch per modality)
   
   - Text: word embeddings passed through a 1D convolution to capture local patterns in the sentence.
   - Audio: a 1D convolution over the waveform or a spectrogram, to pick up pitch, rhythm, and tone.
   - Video: frames from each utterance go through a convolutional backbone, then are combined over time.
   - Image with context: key frames are sampled from the video. The speaker's bounding box is masked out so the model        sees the surroundings, and a depth map is computed with a pretrained monocular depth model. The context features        are combined with the image features by element-wise multiplication, so the scene modulates what the model takes        from the image.

2. Attention layer
   
   - Each modality gets its own attention module, so it can emphasise its own important parts. In parallel, a cross-         attention module looks across all four modalities at once. The two kinds of output are kept as separate streams.

3. Feature decoupling
   
   Each modality's features are split into two parts:

   - Shared part (emotion-aware): produced by an encoder shared across modalities, trained with the emotion labels so        it keeps what is relevant to emotion.
   - Private part (emotion-agnostic): produced by one encoder per modality, trained so that it keeps information about       the input but not about the label. This uses mutual-information estimators (vCLUB and MINE).

4. Graph distillation
   
   - Homogeneous graph distillation (shared features): the four modalities form a directed graph. Each edge has a            learned weight for how much one modality should guide another, and the loss pushes weaker modalities'                   predictions toward more reliable ones.
   - Heterogeneous graph distillation (private features): private features differ a lot between modalities, so a cross-      modality attention step first brings them closer. Distillation then runs with the same kind of weighted loss.

5. Fusion and classification
   
   - For each modality, a learned gate balances its shared and private features. The results are concatenated across         modalities and fed to a classifier.

## Differences from DMD

The shared/private feature split and the graph-distillation framework follow DMD. CAM-DGD differs in these ways:

- **Image-context stream:** DMD works with text, audio, and video. CAM-DGD adds a fourth stream built from sampled frames: images with the speaker masked out, plus depth maps, to represent the scene around the speaker. These context features are combined with the image features.
- **Attention modules:** CAM-DGD adds one attention module per modality, plus a cross-attention module across all four modalities. Their outputs are kept as separate streams.
- **Four-modality graph:** Because of the extra stream, the distillation graph covers four modalities instead of three.

## Dataset

**CMU-MOSEI**: sentence-level video clips with synchronised text, audio, and visual features, annotated for sentiment from -3 to +3 and for six emotions.

## Results

## Results

We evaluate on CMU-MOSEI in the aligned and unaligned settings, reporting 7-class accuracy (ACC7), binary accuracy (ACC2), and F1.

**Aligned setting**

| Method | ACC7 (%) | ACC2 (%) | F1 (%) |
|---|---|---|---|
| EF-LSTM | 47.4 | 78.2 | 77.9 |
| LF-LSTM | 48.8 | 80.6 | 80.6 |
| Graph-MFN | 45.0 | 76.9 | 77.0 |
| RAVEN | 50.0 | 79.1 | 79.5 |
| MCTN | 49.6 | 79.8 | 80.6 |
| MulT | 51.8 | 82.5 | 82.3 |
| PMR | 52.5 | 83.3 | 82.6 |
| DMD | 53.7 | 85.0 | 84.9 |
| MISA | 52.2 | 85.5 | 85.3 |
| FDMER | 54.1 | 86.1 | 85.8 |
| **CAM-DGD (ours)** | **55.3** | **86.7** | **87.1** |

**Unaligned setting**

| Method | ACC7 (%) | ACC2 (%) | F1 (%) |
|---|---|---|---|
| EF-LSTM | 46.3 | 76.1 | 75.9 |
| LF-LSTM | 48.8 | 77.5 | 78.2 |
| RAVEN | 45.5 | 75.4 | 75.7 |
| MCTN | 48.2 | 79.3 | 79.7 |
| MulT | 50.7 | 81.6 | 81.6 |
| PMR | 51.8 | 83.1 | 82.8 |
| DMD | 54.6 | 84.8 | 84.7 |
| **CAM-DGD (ours)** | **55.1** | **85.3** | **85.2** |

## Limitations

- Evaluated on one dataset (CMU-MOSEI), which is mostly single-speaker video, so scene context may carry limited signal.
- Validation and test accuracy plateau early in training while training accuracy keeps rising, even with early  stopping, so the model would likely benefit from stronger regularization or more data.




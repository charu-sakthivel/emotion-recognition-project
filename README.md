# CAM-DGD: Context-Aware Multimodal Decoupled Graph Distillation for Emotion Recognition

CAM-DGD is a multimodal model that predicts emotion and sentiment from text, audio, and video, and also looks at the visual surroundings of the speaker (background and scene), not only the speaker's face and voice.

## Overview

Most multimodal emotion models mix all inputs into one representation. This project tries three ideas instead:

1. Use context: Alongside text, audio, and video, the model gets a fourth stream built from sampled frames: a version of each frame with the speaker masked out, and a depth map of the scene.

2. Separate what is shared from what is specific: For each modality, the model learns one part that is shared across modalities (carrying emotion information) and one part that is private to that modality.

3. Let modalities teach each other: Graph distillation lets modalities pass knowledge to one another, with learned weights deciding who guides whom.

## Architecture





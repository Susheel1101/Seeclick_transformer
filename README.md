# SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents

## Paper Presentation
This repository contains materials for the presentation on the paper "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents" for the Transformers class.

## Authors & Citation
Authors: Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, Zhiyong Wu

```bibtex
@article{cheng2024seeclick,
  title={SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents},
  author={Cheng, Kanzhi and Sun, Qiushi and Chu, Yougang and Xu, Fangzhi and Li, Yantao and Zhang, Jianbing and Wu, Zhiyong},
  journal={arXiv preprint arXiv:2401.10935},
  year={2024}
}
```

## Important Links
1. [Original Paper](https://arxiv.org/abs/2401.10935)
2. [Official Implementation](https://github.com/njucckevin/SeeClick)
3. [Mind2Web: Towards a Generalist Agent for the Web](https://arxiv.org/abs/2306.06070) - Related work on web agents
4. [GPT-4V Technical Report](https://arxiv.org/abs/2303.08774) - Foundation for visual language models
5. [Android in the Wild: A Large-Scale Dataset for Android Device Control](https://arxiv.org/abs/2307.10088) - AITW dataset paper

## Overview
SeeClick is a novel visual GUI agent that operates solely on screenshots to automate tasks across different platforms (mobile, desktop, web). Key innovations include:
- GUI grounding capability for accurate element localization
- Unified vision-language architecture
- ScreenSpot benchmark for evaluation
- State-of-the-art performance with minimal training data

## Technical Architecture
The model comprises three key components:
1. Vision Encoder: ViT for screenshot processing
2. Vision-Language Adapter: Connects visual features to language model
3. Large Language Model: Handles instruction understanding and action generation

## Implementation

### Dependencies
```bash
torch
transformers
matplotlib
pillow
numpy
huggingface_hub
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/[your-username]/SeeClick-Presentation.git
cd SeeClick-Presentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get Hugging Face access token and login:
```python
from huggingface_hub import login
login(token="your_token_here")
```

### Demo
The `demo/seeclick_demo.ipynb` notebook contains a complete implementation demonstrating SeeClick's capability to:
- Load and process screenshots
- Predict element locations based on instructions
- Visualize predictions

## Repository Structure
```
SeeClick-Presentation/
├── README.md
├── demo/
│   ├── seeclick_demo.ipynb
│   └── demo_pictures/
├── resources/
│   └── presentation_content.md
└── requirements.txt
```

## Limitations and Future Work
1. Technical Limitations:
   - Limited action space (basic clicking and typing only)
   - Performance dependency on open-source LVLMs
   - Resolution constraints with high-resolution screenshots

2. Areas for Future Development:
   - Cross-platform optimization
   - Error recovery mechanisms
   - Complex interaction sequences
   - Enhanced grounding for non-text elements

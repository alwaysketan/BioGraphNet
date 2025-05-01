# BioGraphNet: Histopathology Cancer Classification with Graph Neural Networks

![Example Explanation](histopathology_explanation_colorbars.png)

## Overview

BioGraphNet is an advanced deep learning framework for histopathological image analysis that combines convolutional neural networks with graph neural networks. This model achieves state-of-the-art performance in classifying lung and colon cancer subtypes from histopathology images while providing interpretable explanations for its predictions.

Key features:
- Hybrid CNN-GNN architecture for capturing both local and global tissue patterns
- Dynamic graph learning to model complex cellular relationships
- Multi-modal explainability with GNN, feature importance, and LIME visualizations
- Robust training with advanced regularization techniques

## Model Architecture

The architecture consists of three main components:

1. **Residual CNN Backbone**: Extracts hierarchical visual features from histopathology images
2. **Dynamic Graph Learner**: Constructs adaptive graphs from image features to model tissue structures
3. **GATv2-based GNN**: Processes the graph representation with attention mechanisms to capture relational patterns

```
BioGraphNet Architecture:
[Input Image] → [ResNet Backbone] → [Feature Grid]
    ↓
[Graph Construction] → [GATv2 Layers] → [Global Pooling]
    ↓
[Readout Layers] → [Classification]
```

## Dataset

The model was trained on the [Lung and Colon Cancer Histopathological Images dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) containing 25,000 images across 5 classes:

1. Colon Adenocarcinoma (colon_aca)
2. Colon Benign Tissue (colon_n)
3. Lung Adenocarcinoma (lung_aca)
4. Lung Benign Tissue (lung_n)
5. Lung Squamous Cell Carcinoma (lung_scc)

## Performance

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 96.2%  |
| AUC (macro)  | 0.997  |
| Precision    | 95.8%  |
| Recall       | 96.1%  |
| F1-score     | 96.0%  |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BioGraphNet.git
cd BioGraphNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```python
from train import train_and_explain

train_and_explain()
```

### Inference
```python
from explain import HistopathologyExplainer

explainer = HistopathologyExplainer("bio_graph_net.pth")
explainer.explain("path/to/your/image.jpeg")
```

### Explanation Methods
1. **GNN Importance**: Highlights regions most influential in the graph structure
2. **Feature Activation**: Shows which visual features the model focuses on
3. **LIME Superpixels**: Identifies diagnostically relevant regions using local interpretability

## Examples

### Colon Adenocarcinoma
![Colon ACA Example](colon_aca_explanation.png)

### Lung Squamous Cell Carcinoma
![Lung SCC Example](lung_scc_explanation.png)

## Key Files
- `model.py`: Contains the BioGraphNet architecture
- `train.py`: Training and evaluation scripts
- `explain.py`: Explanation generation utilities
- `data.py`: Data loading and preprocessing
- `config.py`: Hyperparameters and settings

## Requirements
- Python 3.8+
- PyTorch 2.2.0+
- PyTorch Geometric
- OpenCV
- scikit-learn
- Lime
- SHAP

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{BioGraphNet2023,
  author = {Your Name},
  title = {BioGraphNet: Interpretable Cancer Classification with Graph Neural Networks},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/BioGraphNet}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, please contact:  
Ketan Dwivedi 
ketandwivedi05@gmail.com 

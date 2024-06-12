# MoSwformer
MoSwformer, a framework for cancer subtype classification, integrating multi-omics data with transfer models, MAL to tackle high dimensionality and complexity. The model leverages attention mechanisms for optimal data weighting and captures commonalities through MAL. The SET encoder enhances pattern recognition, offering a comprehensive cancer subtype analysis approach. This reposiry contains the data and python scripts in support of the manuscript: MoSwformer: A Transformer-based patient classification model using multi-omics data.
# Environment Requirement
The code has been tested running under Python 3.8. The required packages are as follows:
- torch == 1.12.1 (GPU version)
- numpy == 1.23.5
- pandas == 1.5.0
- scikit-learn==1.1.2
# How to run the code
Although we build several *.py* files, running our code is very simple. More specifically, we only need to run *train_test.py* to train the model, outputting prediction results. In addition, running our code requires utilizing PyTorch's deep learning framework under Python 3.8.

# Name Entity Recognetion using LSTM

This repository contains an end-to-end NER pipeline using custom deep learning models implemented in PyTorch. It includes training, inference, and visualization of performance metrics for text classification.

## Project Description

The notebook demonstrates a custom NER classifier trained on preprocessed text data using deep learning. It compares multiple model configurations and evaluates them using accuracy and loss metrics. The training loop and inference routines are implemented from scratch with visualization and interpretability.

## Key Features

- LSTM Models with different configirations 
- Training Loop with Real-time Loss & Accuracy Tracking
- Inference Loop with Sample Predictions

##  Dataset

The dataset used is a NER-labeled dataset conll2003. Each sample consists of a piece of text and its corresponding Entity tags.


## Models Compared

Three different model configurations were trained and compared based on:

- **Training Accuracy**
- **Training Loss**
- **Inference Results**

Each model's performance is visualized using line plots for comparison.

## Usage

### 1. Install Dependencies

```bash
pip install datasets torch matplotlib tqdm
```

### 2. Run the Notebook

Launch the notebook in a Jupyter environment:

```bash
jupyter notebook ner-lstm.ipynb
```

### 3. Results

- Visualizations of training losses and accuracies
- Printed inference samples with predictions vs ground truth
- Final validation accuracy

## 📊 Example Inference Output

```text
[Example 1]
Prediction: ['B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O',
 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
Ground Truth: ['B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 
'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
----------------------------------------
[Example 2]
Prediction: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'B-PER', 'I-PER', 
'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'B-ORG', 'B-ORG', 'B-ORG']
Ground Truth: ['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 
'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
----------------------------------------
[Example 3]
Prediction: ['O', 'I-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
'O', 'O', 'O', 'O', 'O', 'B-ORG', 'B-ORG', 'B-ORG', 'B-ORG', 'B-ORG', 'B-ORG', 'B-ORG', 'B-ORG', 'B-ORG', 'B-ORG', 'B-ORG']
Ground Truth: ['O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
'O', 'B-PER', 'O', 'O', 'O']

 Inference Accuracy: 93.46%
```

## 📈 Training and Evaluation Plots

> Plots include:
> - Loss curves per epoch
> - Accuracy curves per epoch

![result](https://github.com/HeshamEL-Shreif/NER-with-LSTM/blob/main/output.png)


## 🧑‍💻 Author

- Hesham El-Shreif  
- GitHub: [@HeshamEL-Shreif](https://github.com/HeshamEL-Shreif)

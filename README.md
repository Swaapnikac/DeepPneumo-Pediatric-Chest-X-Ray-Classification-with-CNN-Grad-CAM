# DeepPneumo – Pediatric Chest X-Ray Classification with CNN and Grad-CAM  

**Technologies:** Python, PyTorch, CNN, ResNet, Grad-CAM, NumPy, Matplotlib  

---

## Overview  
DeepPneumo is a deep learning–based model developed to classify pediatric chest X-rays as either **Normal** or **Pneumonia**.  
The model uses a **ResNet-based Convolutional Neural Network (CNN)** for feature extraction and classification, with **Grad-CAM visualizations** to interpret which lung regions influenced the final prediction.  

This prototype demonstrates how AI can enhance diagnostic decision support systems in the medical imaging domain.  
The project was completed as part of the **Advanced Medical Device Software Engineering** course at **Northeastern University**.

---

## Objectives  
- Build an efficient CNN-based classifier for pediatric chest X-rays.  
- Implement interpretability through Grad-CAM visualization.  
- Evaluate model performance using accuracy, precision, recall, and F1-score.  
- Demonstrate a reproducible AI workflow suitable for medical research and education.  

---

## Features  
- Automatic pneumonia detection from pediatric chest X-rays.  
- ResNet architecture for deep visual feature extraction.  
- Grad-CAM visualizations for interpretability and transparency.  
- End-to-end notebook with data preprocessing, training, and evaluation.  
- Reproducible training setup with random seed initialization.  

---

## Model Architecture  
Input X-Ray Image → Preprocessing → ResNet Layers → Classification Head → Softmax Output
↳ Grad-CAM Visualization for Explainability

---

## Training and Evaluation Details  

**Model:** ResNet18 (fine-tuned for binary classification)  
**Loss Function:** CrossEntropyLoss  
**Optimizer:** Adam (learning rate = 0.001)  
**Batch Size:** 32  
**Epochs:** 20  
**Image Size:** 224x224 pixels  
**Dataset Split:** 70% Training, 15% Validation, 15% Testing  

**Dataset Reference:**  
Pediatric Chest X-ray Dataset (Kaggle):  
[https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## Performance Metrics  

| Metric | Description | Value |
|:--------|:-------------|:------:|
| Accuracy | Proportion of correct predictions | 92% |
| Precision | True Positives / (True Positives + False Positives) | 90% |
| Recall | True Positives / (True Positives + False Negatives) | 93% |
| F1-Score | Harmonic mean of precision and recall | 91% |
| Validation Loss | Average classification error on validation set | 0.21 |
| Test Loss | Model performance on unseen data | 0.24 |

---

## Example Prediction Output  

**Predicted:** Pneumonia  
**Actual:** Pneumonia  
**Confidence:** 0.94  

Grad-CAM visualization highlights the lung regions with dense opacities that influenced the classification, enhancing the interpretability of the CNN’s decision-making process.

---

## Results Summary  

- The model achieved **92% test accuracy**, demonstrating strong generalization performance.  
- Grad-CAM heatmaps confirmed that the CNN focused on the lung regions while identifying pneumonia cases.  
- Training and validation loss curves showed consistent convergence after 15 epochs, indicating a well-regularized model.  
- Evaluation metrics such as precision, recall, and F1-score reflect reliable classification on medical imaging data.  

---

## Applications  
- AI-assisted radiology for pneumonia screening.  
- Educational research in medical imaging and AI explainability.  
- Prototyping diagnostic decision support systems for healthcare.  
- Developing interpretable AI tools for clinical transparency.  

---

## Future Enhancements  
- Integration with a Streamlit or Flask web interface for real-time image uploads.  
- Deployment on platforms like Hugging Face Spaces or Gradio for live demonstrations.  
- Expansion to multi-class classification (e.g., bacterial vs. viral pneumonia).  
- Incorporation of uncertainty quantification to enhance model reliability in clinical settings.  

---

## Author  
**Cherukuru Swaapnika Chowdary**  
Graduate Student, Northeastern University  
Boston, Massachusetts, USA  
https://www.linkedin.com/in/swaapnika-cherukuru-926990228/

---

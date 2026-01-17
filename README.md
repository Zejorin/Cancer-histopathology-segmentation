**Multi-Class Histopathology Segmentation \& Classification**

This repository implements an advanced medical AI pipeline using MONAI and SegResNet to perform multi-class tissue segmentation and classification on histopathology slides. It marks an evolution from simple binary models to a complex 6-class architecture capable of distinguishing between high-priority biological features.



**Project Overview**

The primary challenge of this project was the transition from a binary "cell vs. background" structure to a 6-class tissue classifier. This shift required a complete architectural redesign, moving from Sigmoid-based binary logic to a Softmax/Argmax pipeline to handle competing cell categories including:

Neoplastic (Cancer)

Inflammatory

Epithelial

Connective/Soft Tissue

Dead Cells

Background



***Key Results***

Segmentation (DICE Score): 0.68

Classification (F1-Score): 0.71

Architecture: SegResNet (2D) with init\_filters=32 to capture complex morphological features.

Loss Strategy: Weighted DiceCELoss (Weighted 4.0 for Neoplastic cells to prioritize cancer detection).



**Methodology**

1. Data Engineering \& Integrity
   Master List Synchronization: Implemented a robust metadata tracking system that "glues" tissue-type labels (Breast, Lung, etc.) and original indices directly to image/mask dictionaries.

Stratification: Data split into 80/10/10 (Train/Val/Test) to maintain class proportions.

2. Model Evolution
   SegResNet: Upgraded the baseline model to 32 initial filters, significantly increasing the model's capacity to distinguish between similar-looking cell types.

Multi-Class Pipeline: Transitioned to a Softmax activation and Argmax post-processing flow to ensure pixels are assigned to unique, competing classes.

3. Optimized Loss Function
   DiceCELoss: Combined Dice Loss for spatial accuracy with Cross-Entropy for class probability.
4. Class Weighting: Applied manual weights to the Cross-Entropy component to address class imbalance, specifically penalizing the model more heavily for missing Neoplastic cells.



**Environment Setup**
Ensure you have Python 3.10+ and the required medical imaging libraries: pip install torch torchvision monai numpy matplotlib tqdm scikit-learn kaggle



**Data Organization:**
After running import.py, your folder should look like this:

📁 your-repo-name

├── 📁 .venv (if running virtal environment)

├── 📁 Author notes

├── 📁 Images

├── 📁 Masks

├── {} dataset-metadata.json

├── 📁 dataset

│   ├── images.npy

│   ├── masks.npy

└── segmentation.py

└── ImportData.py



**Performance Analysis \& Challenges**


GPU Memory Management: Increasing filter density to 32 required careful management of GPU VRAM, leading to the implementation of smaller batch sizes and specialized memory cleanup routines to prevent CUDNN execution failures.

Interpretability: Developed a custom show\_prediction visualization tool that synchronizes RGB images with 6-channel ground truth and predicted masks for clinical auditability.



False Negatives: While the model shows strong performance, it still faces challenges where neoplastic cells are occasionally predicted as non-cancerous, highlighting the need for further fine-tuning in digital pathology contexts.



**Future Directions \& Feedback**


I am always looking to refine my approach to medical imaging. I welcome feedback on:

Handling extreme class imbalance in histopathology.

Optimizing MONAI transforms for better Dice score performance.

If you would like to discuss the codebase or technical implementation, please feel free to reach out!




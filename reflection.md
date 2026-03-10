# Brain Tumor Detection Using VGG16 – Reflection

## Challenges Faced

1. **Dataset Acquisition**: Obtaining labeled MRI brain images required searching for publicly available datasets. The Kaggle "Brain MRI Images for Brain Tumor Detection" dataset uses `yes`/`no` folder naming instead of `tumor`/`normal`, requiring careful mapping during preprocessing.

2. **Computational Resources**: Training VGG16 (138M parameters) is computationally expensive. Without a GPU, training even 5 epochs takes significant time. This motivated the use of transfer learning with pretrained ImageNet weights rather than training from scratch.

3. **Overfitting Risk**: The dataset is relatively small (~250 images). Data augmentation (random horizontal flip, rotation) and using pretrained weights helped mitigate overfitting, but the model can still memorize training data with extended training.

4. **Learning Rate Sensitivity**: Experiment 1 showed that the learning rate significantly impacts convergence speed and final accuracy. A higher learning rate (0.001) converges faster but may overshoot, while 0.0001 trains more slowly but can be more stable.

5. **Frozen vs. Unfrozen Layers**: Freezing the convolutional feature layers and only training the classifier reduces training time dramatically but can limit the model's ability to adapt to domain-specific MRI features that differ from ImageNet images.

6. **Image Quality Variability**: MRI images vary in contrast, orientation, and resolution across different sources, which can reduce model generalization unless proper normalization and augmentation are applied.

## Key Takeaways

- Transfer learning is essential for medical imaging tasks with limited labeled data.
- VGG16's deep feature extraction layers effectively capture relevant patterns in brain MRI scans even though they were trained on natural images.
- Careful hyperparameter tuning (learning rate, epochs) and architectural decisions (frozen vs. unfrozen layers) meaningfully impact model performance.
- A confusion matrix provides more insight than accuracy alone—false negatives (missed tumors) are more critical than false positives in a medical context.

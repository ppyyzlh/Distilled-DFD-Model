# Distilled-DFD-Model
deepfake detection

$$

\text{distill_loss}(output, teacher_output, labels) &= \frac{1}{n}\sum_{i=1}^n \left[ \frac{T^2}{c} \sum_{j=1}^c \frac{\exp(teacher\_output_{ij}/T)}{\sum_{k=1}^c \exp(teacher\_output_{ik}/T)} \log \frac{\exp(output_{ij}/T)}{\sum_{k=1}^c \exp(output_{ik}/T)} \right. \\
&\left. + (1-\alpha) (-\log \frac{\exp(output_{il_i})}{\sum_{k=1}^c \exp(output_{ik})}) \right]

$$
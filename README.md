# ConvLSTM Video Prediction: Next-Frame Forecasting on Moving MNIST

## Abstract
This project implements a **ConvLSTM (Convolutional Long Short-Term Memory)** neural network for the task of Spatiotemporal Video Prediction. The objective is to predict the subsequent frames in a video sequence given a history of consecutive frames. The model is trained on the **Moving MNIST** dataset, learning to model both the spatial appearance (digits) and the temporal dynamics (movement trajectories) simultaneously.

## Methodology

### Problem Definition
Given a sequence of frames $X_{t-n}, ..., X_{t}$, the goal is to predict the next frame $\hat{Y}_{t+1}$. This is treated as a self-supervised learning problem where the ground truth is the actual future frame in the sequence.

### Algorithm: ConvLSTM
Standard LSTM networks process 1D vectors, which necessitates flattening image data, leading to a loss of spatial structural information. **ConvLSTM** overcomes this by replacing the internal matrix multiplications with convolution operations. This allows the flow of data through the LSTM cells to maintain 5D tensors: `(Batch, Time, Height, Width, Channels)`.

The core equation for the ConvLSTM cell is:

$$
\begin{aligned}
i_t &= \sigma(W_{xi} * \mathcal{X}_t + W_{hi} * \mathcal{H}_{t-1} + W_{ci} \circ \mathcal{C}_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} * \mathcal{X}_t + W_{hf} * \mathcal{H}_{t-1} + W_{cf} \circ \mathcal{C}_{t-1} + b_f) \\
\mathcal{C}_t &= f_t \circ \mathcal{C}_{t-1} + \tanh(W_{xc} * \mathcal{X}_t + W_{hc} * \mathcal{H}_{t-1} + b_c) \\
o_t &= \sigma(W_{xo} * \mathcal{X}_t + W_{ho} * \mathcal{H}_{t-1} + W_{co} \circ \mathcal{C}_t + b_o) \\
\mathcal{H}_t &= o_t \circ \tanh(\mathcal{C}_t)
\end{aligned}
$$

Where $*$ denotes the convolution operator and $\circ$ denotes the Hadamard product.

## Model Architecture
The implemented architecture follows a deep Encoder-Decoder structure designed to capture multi-scale spatiotemporal features.

1.  **Input Layer:** `(Batch, 19, 64, 64, 1)` - Sequence of 19 grayscale frames.
2.  **ConvLSTM Layer 1:** 64 Filters, 5x5 Kernel. Captures broad motion patterns.
    *   *Batch Normalization + ReLU*
3.  **ConvLSTM Layer 2:** 64 Filters, 3x3 Kernel. Refines local spatial details and object interactions.
    *   *Batch Normalization + ReLU*
4.  **ConvLSTM Layer 3:** 64 Filters, 1x1 Kernel. Acts as a feature bottleneck and mixer.
    *   *Batch Normalization + ReLU*
5.  **Output Layer (Conv3D):** 3x3x3 Kernel with Sigmoid Activation. Projects the features back to the pixel space `[0, 1]`.

## Loss Functions
To ensure high fidelity and avoid blurriness common in MSE-based video prediction, a composite loss function is employed:

*   **Binary Cross Entropy (BCE):** Ensures pixel-level accuracy.
*   **Perceptual Loss (VGG16):** Minimizes the difference in high-level feature representations, resulting in more natural-looking digits.
*   **Gradient Difference Loss (GDL):** Penalizes differences in image gradients to sharpen edges and maintain structural integrity.

## Results

The model performance is evaluated using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Mean Squared Error (MSE).

| Metric | Average Value | Description |
| :--- | :--- | :--- |
| **PSNR** | **~25.0 dB** | Measures reconstruction quality. Higher is better. |
| **SSIM** | **0.85** | Measures structural similarity (0 to 1). Closer to 1 is better. |
| **MSE** | **Low** | Measures the average squared difference between estimated and actual values. |

### Visual Analysis

The figure below compares the Ground Truth (Top Row) with the Model Predictions (Bottom Row). The model successfully predicts the trajectory and preserves the shape of the digits.

![Prediction Comparison](./results/evaluation_results.png)

### Training Convergence

The training loss curve demonstrates stable convergence, indicating effective optimization of the composite loss function.

![Training Loss](./results/loss_curve.png)

## Installation and Usage

**Prerequisites:** Python 3.8+, TensorFlow 2.10+

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/samettkartal/ConvLSTM-Video-Prediction.git
    cd ConvLSTM-Video-Prediction
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model**
    ```bash
    python train.py
    ```

4.  **Evaluate**
    ```bash
    python evaluate.py
    ```

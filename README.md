# BaybGPT




BaybGPT is a simple Decoder-Only Transformer model implemented in PyTorch. This project serves as a learning exercise to understand the inner workings of Decoder-Only Transformer architectures.

## Introduction to Decoder-Only Transformer
A **decoder-only Transformer** is a type of neural network architecture designed for autoregressive text generation. Unlike traditional Transformer models that have both an encoder and a decoder (such as in machine translation), a decoder-only Transformer consists of only the decoder blocks.  

### Core Components:  
- **Self-Attention Mechanism:** Uses multi-head self-attention to process input tokens in parallel while maintaining dependencies between words.  
- **Causal Masking:** Ensures that each token can only attend to previous tokens, preventing it from seeing future information.  
- **Feed-Forward Networks (FFN):** Each transformer block includes a fully connected feed-forward network that enhances learning capacity.  
- **Dropout Regularization:** Helps prevent overfitting by randomly deactivating a fraction of neurons during training.  
- **Residual Connections:** Allows gradients to flow more effectively during backpropagation, improving training stability.  
- **Layer Normalization:** Stabilizes training and improves convergence speed.  


## Usage
1. **Configuration:**  
    Edit the `config.ini` and the `run.sh` files to set your model, training, and dataset parameters.

2. **Run Code:**  
    Run the training script:
    ```bash
    bash run.sh
    ```
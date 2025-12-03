# Machine-learning-Home-Assignment-5


This code implements two fundamental components of the Transformer neural network architecture.

Q1 (NumPy) implements the Scaled Dot-Product Attention mechanism, calculating alignment scores between Query (Q) and Key (K) matrices, scaling them by 1/dk​​ for stability, and applying softmax to derive Attention Weights. These weights are then multiplied by the Value (V) matrix to produce the final Context Vector.

Q2 (PyTorch) defines the Transformer Encoder Block. This class integrates a Multi-Head Attention layer, a Feed-Forward Network (FFN), and the essential Add & Norm steps (residual connections followed by Layer Normalization) to process input embeddings sequentially. It verifies that input and output tensor shapes remain consistent, a crucial check for building deep networks.

# LSTM & Recurrent Model Experiment Results

Evaluation on Low-Quality Auslan Dataset (70/30 stratified train/test split, random seed 42).

### Reference Benchmarks
- **Paper's LSTM (MSE loss, step-by-step)**: Macro F1 = `0.066`
- **Our RBF SVM (C=10, flattened 456-dim features)**: Accuracy = `0.590`, Macro F1 = `0.589`
- **Paper's SVM baseline**: Macro F1 = `0.549`

### Systematic Experiment Suite

| Experiment | Scaling | Architecture | Hidden Units | Dropout | Stacked | Stopped Epoch | Test Accuracy | Macro F1 |
|---|---|---|---|---|---|---|---|---|
| Baseline LSTM (128) | per-example | LSTM | 128 | 0.3 | False | 38 | **0.383** | **0.370** |
| Hidden 64 | per-example | LSTM | 64 | 0.3 | False | 60 | **0.350** | **0.339** |
| Hidden 256 | per-example | LSTM | 256 | 0.3 | False | 26 | **0.409** | **0.396** |
| GRU (128) | per-example | GRU | 128 | 0.3 | False | 35 | **0.334** | **0.315** |
| Bidirectional LSTM (128) | per-example | BILSTM | 128 | 0.3 | False | 27 | **0.437** | **0.431** |
| No Dropout (p=0.0) | per-example | LSTM | 128 | 0.0 | False | 26 | **0.347** | **0.332** |
| Stacked LSTM (2x128) | per-example | LSTM | 128 | 0.3 | True | 36 | **0.447** | **0.440** |
| Global Scaling | global | LSTM | 128 | 0.3 | False | 55 | **0.380** | **0.368** |

### Key Observations & Discussion
1. **Resolution of the Paper's LSTM Failure**: The paper reported an F1 of 0.066, attributing failure to backpropagating at every time step with MSE loss. By computing crossentropy loss exclusively from the final time step (`return_sequences=False`) and using softmax across the 95 classes, performance increases by an order of magnitude.
2. **Comparison with RBF SVM**: While recurrent architectures capture sequential temporal dynamics, RBF SVM on the flattened 456-dimensional representation remains competitive or superior on this small dataset (~4,650 training sequences across 95 classes, i.e., ~49 examples per class). Deep neural networks are data-hungry and prone to overfitting with such few samples per class.
3. **Architectural Variations**: GRU, Bidirectional LSTM, and stacked layers explore different temporal inductive biases. Bidirectional modeling allows the network to incorporate both onset and ending hand configurations.

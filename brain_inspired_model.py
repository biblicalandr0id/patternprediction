"""
Brain Inspired Model – Production‑Ready Implementation

This model attempts to emulate certain aspects of biological brain architecture:
- Recursive, nested (wrapped) layers for hierarchical processing
- Feedback connections where the output from inner layers is fed back to
  outer layers for refinement
- Unique parameters per layer while also incorporating cross-layer communication
- Cutting‑edge techniques such as layer normalization, residual connections, and dropout

This design is intended as a conceptual exploration of a brain‑inspired model,
and it can be extended or refined further for production as needed.

Usage:
  python brain_inspired_model.py
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

class BrainInspiredModel:
    def __init__(self, vocab_size, hidden_dim, num_layers, dropout=0.2, learning_rate=0.001):
        self.vocab_size = vocab_size      # e.g., 256 for full ASCII mapping
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers        # Number of functional modules/layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Initialize unique LSTM parameters for each layer.
        self.params = {}
        # For the very first layer, input dimension is vocab_size (one-hot encoding)
        layer_input_dim = self.vocab_size
        for l in range(num_layers):
            # Input gate parameters
            self.params[f"Wxi_{l}"] = np.random.randn(hidden_dim, layer_input_dim) * 0.01
            self.params[f"Uxi_{l}"] = np.random.randn(hidden_dim, hidden_dim) * 0.01
            self.params[f"bi_{l}"]  = np.zeros((hidden_dim, 1))
            # Forget gate parameters
            self.params[f"Wxf_{l}"] = np.random.randn(hidden_dim, layer_input_dim) * 0.01
            self.params[f"Uxf_{l}"] = np.random.randn(hidden_dim, hidden_dim) * 0.01
            self.params[f"bf_{l}"]  = np.zeros((hidden_dim, 1))
            # Output gate parameters
            self.params[f"Wxo_{l}"] = np.random.randn(hidden_dim, layer_input_dim) * 0.01
            self.params[f"Uxo_{l}"] = np.random.randn(hidden_dim, hidden_dim) * 0.01
            self.params[f"bo_{l}"]  = np.zeros((hidden_dim, 1))
            # Cell candidate parameters
            self.params[f"Wxc_{l}"] = np.random.randn(hidden_dim, layer_input_dim) * 0.01
            self.params[f"Uxc_{l}"] = np.random.randn(hidden_dim, hidden_dim) * 0.01
            self.params[f"bc_{l}"]  = np.zeros((hidden_dim, 1))
            # For subsequent layers the input dimension equals the hidden_dim
            layer_input_dim = hidden_dim

        # Output layer: Map the final wrapped activation to vocabulary logits.
        self.params["Why"] = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.params["by"] = np.zeros((vocab_size, 1))

    def lstm_cell_forward(self, x, h_prev, c_prev, layer):
        """
        Single LSTM cell forward pass for a given layer.
        """
        Wxi = self.params[f"Wxi_{layer}"]
        Uxi = self.params[f"Uxi_{layer}"]
        bi  = self.params[f"bi_{layer}"]
        Wxf = self.params[f"Wxf_{layer}"]
        Uxf = self.params[f"Uxf_{layer}"]
        bf  = self.params[f"bf_{layer}"]
        Wxo = self.params[f"Wxo_{layer}"]
        Uxo = self.params[f"Uxo_{layer}"]
        bo  = self.params[f"bo_{layer}"]
        Wxc = self.params[f"Wxc_{layer}"]
        Uxc = self.params[f"Uxc_{layer}"]
        bc  = self.params[f"bc_{layer}"]

        i_gate = sigmoid(np.dot(Wxi, x) + np.dot(Uxi, h_prev) + bi)
        f_gate = sigmoid(np.dot(Wxf, x) + np.dot(Uxf, h_prev) + bf)
        o_gate = sigmoid(np.dot(Wxo, x) + np.dot(Uxo, h_prev) + bo)
        c_candidate = np.tanh(np.dot(Wxc, x) + np.dot(Uxc, h_prev) + bc)

        c = f_gate * c_prev + i_gate * c_candidate
        h = o_gate * np.tanh(c)
        return h, c

    def forward_recursive(self, x, layer, h_prev_layers, c_prev_layers):
        """
        Recursively compute the forward pass in a wrapped manner.
        - Each layer processes its input using an LSTM cell.
        - If not the innermost layer, the layer calls the next inner layer using its activation.
        - Feedback (from inner layers) is combined with the current layer's normalized activation via a residual connection.
        - This process mimics modulatory feedback found in biological brains.
        Returns:
          wrapped_output: the combined output from the current layer and feedback from inner layers.
          h: current layer's hidden state.
          c: current layer's cell state.
        """
        h_prev = h_prev_layers[layer]
        c_prev = c_prev_layers[layer]

        # Compute current layer's LSTM output.
        h, c = self.lstm_cell_forward(x, h_prev, c_prev, layer)
        h_norm = layer_norm(h)
        
        # Optionally apply dropout.
        if self.dropout > 0:
            dropout_mask = (np.random.rand(*h_norm.shape) > self.dropout) / (1 - self.dropout)
            h_norm = h_norm * dropout_mask
        
        # If not the innermost layer, recursively process inner layers.
        if layer < self.num_layers - 1:
            inner_output, inner_h, inner_c = self.forward_recursive(h_norm, layer + 1, h_prev_layers, c_prev_layers)
            # Combine current layer activation with feedback.
            wrapped_output = layer_norm(h_norm + inner_output)
        else:
            wrapped_output = h_norm

        return wrapped_output, h, c

    def forward(self, inputs, h_prev_layers, c_prev_layers):
        """
        Process a sequence of input indices through the brain‑inspired model.
        inputs: list of integers (e.g., character indices)
        h_prev_layers, c_prev_layers: lists of initial hidden and cell states (one per layer)
        Returns:
          ps: dictionary mapping time step to output probability distributions
          cache: (xs, hs, cs, inputs) for backprop if required
        """
        T = len(inputs)
        xs, ps = {}, {}
        hs, cs = {}, {}

        # Initialize previous states (time step -1)
        hs[-1] = {l: np.copy(h_prev_layers[l]) for l in range(self.num_layers)}
        cs[-1] = {l: np.copy(c_prev_layers[l]) for l in range(self.num_layers)}

        for t in range(T):
            # One-hot encode current input.
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1

            # Begin recursive processing at the outermost layer.
            wrapped_output, h_out, c_out = self.forward_recursive(xs[t], 0, hs[t-1], cs[t-1])
            
            # For simplicity, set outer layer states for all layers with h_out (advanced implementations could store each layer separately).
            hs[t] = {l: h_out for l in range(self.num_layers)}
            cs[t] = {l: c_out for l in range(self.num_layers)}
            
            # Map the wrapped output to vocabulary logits.
            logits = np.dot(self.params["Why"], wrapped_output) + self.params["by"]
            exp_logits = np.exp(logits - np.max(logits))
            ps[t] = exp_logits / np.sum(exp_logits)

        cache = (xs, hs, cs, inputs)
        return ps, cache

    def sample(self, h_prev_layers, c_prev_layers, seed_ix, n):
        """
        Generate a sequence of n indices (e.g., characters) from the model.
        h_prev_layers, c_prev_layers: initial hidden and cell states (lists for each layer)
        seed_ix: initial input (integer index)
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        h = [np.copy(h) for h in h_prev_layers]
        c = [np.copy(c) for c in c_prev_layers]
        indices = []
        for t in range(n):
            wrapped_output, h_out, c_out = self.forward_recursive(x, 0, h, c)
            logits = np.dot(self.params["Why"], wrapped_output) + self.params["by"]
            exp_logits = np.exp(logits - np.max(logits))
            p = exp_logits / np.sum(exp_logits)
            ix = np.random.choice(self.vocab_size, p=p.ravel())
            indices.append(ix)
            # Prepare next input as one-hot encoded vector.
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            # Update states for next time step.
            for l in range(self.num_layers):
                h[l] = h_out
                c[l] = c_out
        return indices

def encode_text(text, max_len):
    encoded = [ord(c) % 256 for c in text]
    if len(encoded) < max_len:
        return encoded + [0]*(max_len - len(encoded))
    return encoded[:max_len]

def decode_text(indices):
    return "".join(chr(i) for i in indices if i != 0)

if __name__ == "__main__":
    # Configuration parameters.
    vocab_size = 256
    hidden_dim = 512
    num_layers = 4  # Number of wrapped layers
    seq_length = 50  # For training (not used in sampling here)

    # Instantiate the brain-inspired model.
    model = BrainInspiredModel(vocab_size, hidden_dim, num_layers, dropout=0.1)
    seed_index = np.random.randint(0, vocab_size)
    print("Using seed index:", seed_index)

    # Initialize hidden and cell states for each layer.
    h0 = [np.zeros((hidden_dim, 1)) for _ in range(num_layers)]
    c0 = [np.zeros((hidden_dim, 1)) for _ in range(num_layers)]

    # Generate a sequence of 100 characters.
    generated_indices = model.sample(h0, c0, seed_index, 100)
    generated_text = decode_text(generated_indices)
    print("Generated text:")
    print(generated_text)
```` ▋
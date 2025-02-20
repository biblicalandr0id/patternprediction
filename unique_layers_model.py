"""
Unique Layers Model – Production‑Ready Implementation with Cutting‑Edge Techniques

This module implements an advanced LSTM‑based model with unique parameters in each layer.
Cutting‑edge techniques include:
  - Unique parameters per layer
  - Layer Normalization after each layer’s output for stability
  - Residual connections (when input and output dimensions match)
  - Dropout for regularization

The model is designed for tasks like text generation. Run the file directly to see a demonstration
or import the module to integrate with your production pipeline.

Usage:
  python unique_layers_model.py
"""

import numpy as np

# Configuration settings for the model.
config = {
    "vocab_size": 256,        # e.g. full ASCII mapping
    "hidden_dim": 512,        # Dimensionality of the hidden state
    "seq_length": 50,         # Training sequence length (if applicable)
    "num_layers": 4,          # Number of LSTM layers (unique per layer)
    "dropout": 0.2,           # Dropout probability between layers
    "learning_rate": 0.001,   # Learning rate for parameter updates
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def layer_norm(x, eps=1e-5):
    # Layer normalization: normalize activations across features
    mean = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

class UniqueLayersLSTM:
    def __init__(self, config):
        self.vocab_size = config["vocab_size"]
        self.hidden_dim = config["hidden_dim"]
        self.seq_length = config["seq_length"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.learning_rate = config["learning_rate"]

        # Initialize unique parameters for each LSTM layer.
        self.params = {}
        layer_input_dim = self.vocab_size  # first layer takes one-hot vector input
        for l in range(self.num_layers):
            self.params[f"Wxi_{l}"] = np.random.randn(self.hidden_dim, layer_input_dim) * 0.01
            self.params[f"Uxi_{l}"] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
            self.params[f"bi_{l}"]  = np.zeros((self.hidden_dim, 1))
            
            self.params[f"Wxf_{l}"] = np.random.randn(self.hidden_dim, layer_input_dim) * 0.01
            self.params[f"Uxf_{l}"] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
            self.params[f"bf_{l}"]  = np.zeros((self.hidden_dim, 1))
            
            self.params[f"Wxo_{l}"] = np.random.randn(self.hidden_dim, layer_input_dim) * 0.01
            self.params[f"Uxo_{l}"] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
            self.params[f"bo_{l}"]  = np.zeros((self.hidden_dim, 1))
            
            self.params[f"Wxc_{l}"] = np.random.randn(self.hidden_dim, layer_input_dim) * 0.01
            self.params[f"Uxc_{l}"] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
            self.params[f"bc_{l}"]  = np.zeros((self.hidden_dim, 1))
            
            # After the first layer, input dimension equals hidden_dim
            layer_input_dim = self.hidden_dim

        # Output layer: Map the final hidden state to vocabulary logits.
        self.params["Why"] = np.random.randn(self.vocab_size, self.hidden_dim) * 0.01
        self.params["by"] = np.zeros((self.vocab_size, 1))

    def lstm_cell_forward(self, x, h_prev, c_prev, layer):
        # Obtain parameters for layer 'layer'
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

        # Compute gates
        i_gate = sigmoid(np.dot(Wxi, x) + np.dot(Uxi, h_prev) + bi)
        f_gate = sigmoid(np.dot(Wxf, x) + np.dot(Uxf, h_prev) + bf)
        o_gate = sigmoid(np.dot(Wxo, x) + np.dot(Uxo, h_prev) + bo)
        c_candidate = np.tanh(np.dot(Wxc, x) + np.dot(Uxc, h_prev) + bc)

        c = f_gate * c_prev + i_gate * c_candidate
        h = o_gate * np.tanh(c)
        return h, c

    def forward(self, inputs, h_prev_layers, c_prev_layers):
        """
        Perform a forward pass for an input sequence.
        inputs: list of integers (indices)
        h_prev_layers: previous hidden states for each layer (list of np.array)
        c_prev_layers: previous cell states for each layer (list of np.array)
        Returns:
          - probabilities for each time step
          - cache containing intermediates (if needed for training)
        """
        T = len(inputs)
        xs, hs, cs = {}, {}, {}
        hs[-1] = {}
        cs[-1] = {}
        for l in range(self.num_layers):
            hs[-1][l] = np.copy(h_prev_layers[l])
            cs[-1][l] = np.copy(c_prev_layers[l])
        
        ps = {}
        for t in range(T):
            # One-hot encoding of input character
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            hs[t] = {}
            cs[t] = {}
            a = xs[t]
            for l in range(self.num_layers):
                # Forward step for each layer with unique parameters.
                h_prev = hs[t-1][l] if t > 0 else hs[-1][l]
                c_prev = cs[t-1][l] if t > 0 else cs[-1][l]
                h_cur, c_cur = self.lstm_cell_forward(a, h_prev, c_prev, l)
                
                # Apply layer normalization to the hidden state.
                h_norm = layer_norm(h_cur)
                
                # Use a residual connection if dimensions match.
                if a.shape == h_norm.shape:
                    h_out = h_norm + a
                else:
                    h_out = h_norm
                
                # Optionally apply dropout on the layer output.
                if self.dropout > 0 and l < self.num_layers - 1:
                    dropout_mask = (np.random.rand(*h_out.shape) > self.dropout) / (1 - self.dropout)
                    h_out = h_out * dropout_mask
                
                hs[t][l] = h_out
                cs[t][l] = c_cur
                # The output of current layer becomes input to the next layer.
                a = h_out

            # Compute logits and probability for current time step.
            logits = np.dot(self.params["Why"], a) + self.params["by"]
            exp_logits = np.exp(logits - np.max(logits))
            ps[t] = exp_logits / np.sum(exp_logits)
        cache = (xs, hs, cs, inputs)
        return ps, cache

    def lossFun(self, inputs, targets, h_prev_layers, c_prev_layers):
        T = len(inputs)
        ps, cache = self.forward(inputs, h_prev_layers, c_prev_layers)
        loss = 0
        for t in range(T):
            loss += -np.log(ps[t][targets[t], 0] + 1e-8)
        return loss, cache

    def sample(self, h_prev_layers, c_prev_layers, seed_ix, n):
        """
        Generates a sequence of n indices from the model.
        h_prev_layers, c_prev_layers: initial hidden and cell states (lists for each layer)
        seed_ix: initial input (integer index)
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        h = [np.copy(h) for h in h_prev_layers]
        c = [np.copy(c) for c in c_prev_layers]
        indices = []
        for t in range(n):
            a = x
            for l in range(self.num_layers):
                h_prev = h[l]
                c_prev = c[l]
                h_cur, c_cur = self.lstm_cell_forward(a, h_prev, c_prev, l)
                h_norm = layer_norm(h_cur)
                if a.shape == h_norm.shape:
                    h_out = h_norm + a
                else:
                    h_out = h_norm
                if self.dropout > 0 and l < self.num_layers - 1:
                    dropout_mask = (np.random.rand(*h_out.shape) > self.dropout) / (1 - self.dropout)
                    h_out = h_out * dropout_mask
                h[l] = h_out
                c[l] = c_cur
                a = h_out
            logits = np.dot(self.params["Why"], a) + self.params["by"]
            exp_logits = np.exp(logits - np.max(logits))
            p = exp_logits / np.sum(exp_logits)
            ix = np.random.choice(self.vocab_size, p=p.ravel())
            indices.append(ix)
            # Prepare next input
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
        return indices

# Utility functions for text encoding and decoding
def encode_text(text, max_len):
    encoded = [ord(c) % 256 for c in text]
    if len(encoded) < max_len:
        return encoded + [0]*(max_len - len(encoded))
    else:
        return encoded[:max_len]

def decode_text(indices):
    return "".join(chr(i) for i in indices if i != 0)

# Example usage of the unique layers LSTM model.
if __name__ == "__main__":
    model = UniqueLayersLSTM(config)
    
    # Use a random seed index to generate text.
    seed_index = np.random.randint(0, config["vocab_size"])
    print("Using seed index:", seed_index)
    
    # Initialize hidden and cell states for each layer.
    h0 = [np.zeros((config["hidden_dim"], 1)) for _ in range(config["num_layers"])]
    c0 = [np.zeros((config["hidden_dim"], 1)) for _ in range(config["num_layers"])]
    
    # Generate a sequence of 100 characters.
    generated_indices = model.sample(h0, c0, seed_index, 100)
    generated_text = decode_text(generated_indices)
    
    print("Generated text:")
    print(generated_text)
```` ▋
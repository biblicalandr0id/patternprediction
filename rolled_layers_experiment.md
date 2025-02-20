```markdown
# Fully Rolled Layers: Shared Transformation Across Layers

In a traditional deep neural network, each layer has its own unique parameters that enable the model to capture different levels of feature abstraction. An alternative design is to *roll* all layers—using the same set of parameters across all layers. This weight‑sharing or "rolled" architecture applies a single transformation repeatedly to simulate a deep stack.

## Advantages
- **Parameter Efficiency:** Using a single set of parameters for multiple layers reduces the overall parameter count.
- **Consistent Transformation:** Enforces a consistent transformation at every stage, which may enhance generalization.
- **Reduced Memory Usage:** Fewer parameters means less memory consumption during training and inference.

## Disadvantages
- **Limited Representational Diversity:** Because every "layer" is identical, the model may not capture a wide diversity of features as well as a truly deep network.
- **Reduced Flexibility:** The same transformation is applied at every iteration, which might hinder its ability to model very complex relationships.

## Production‑Ready Implementation

Below is a fully self‑contained Python file that implements a **Fully Rolled Model**. The model uses a single transformation (with a weight matrix and bias) that is applied multiple times—i.e. "rolled out" across the same number of layers that a traditional deep network would employ. The code includes functions for forward propagation, prediction, sampling text, and an example usage.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FullyRolledModel:
    def __init__(self, vocab_size, hidden_dim, num_rolls, learning_rate=0.001):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_rolls = num_rolls  # The number of times the single layer is rolled out
        self.learning_rate = learning_rate
        
        # Shared transformation parameters (rolled over all layers)
        self.W = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b = np.zeros((hidden_dim, 1))
        
        # Output layer parameters: mapping from hidden state to vocabulary logits
        self.Why = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.by = np.zeros((vocab_size, 1))
    
    def forward_roll(self, h):
        """
        Applies the shared transformation repeatedly for a specified
        number of roll-out iterations.
        """
        for _ in range(self.num_rolls):
            h = sigmoid(np.dot(self.W, h) + self.b)
        return h
    
    def predict(self, h):
        """
        Runs the forward pass on an initial hidden state `h` through the rolled layer,
        then computes output logits and probabilities.
        """
        h_final = self.forward_roll(h)
        y = np.dot(self.Why, h_final) + self.by
        exp_y = np.exp(y - np.max(y))
        return exp_y / np.sum(exp_y)
    
    def sample(self, seed_ix, n):
        """
        Generates a sequence of `n` indices (e.g., characters) starting from a seed index.
        A one-hot encoded vector is created from the seed, and the transformation is rolled out.
        """
        # Create an initial hidden state using one-hot encoding (for demonstration)
        h = np.zeros((self.hidden_dim, 1))
        h[seed_ix % self.hidden_dim] = 1
        
        indices = []
        for _ in range(n):
            p = self.predict(h)
            ix = np.random.choice(self.vocab_size, p=p.ravel())
            indices.append(ix)
            # Optionally update state: here we roll the same h forward
            h = self.forward_roll(h)
        return indices

# Example usage: using fully rolled layers to generate text.
if __name__ == "__main__":
    # Configuration parameters
    vocab_size = 256      # e.g. full ASCII/extended mapping
    hidden_dim = 512      # Dimensionality of the hidden state
    num_rolls = 4         # Equivalent to the number of layers in a deep network
    num_chars = 100       # Number of characters to generate
    
    # Instantiate the model
    model = FullyRolledModel(vocab_size, hidden_dim, num_rolls)
    
    # Use a random seed index for generation (simulate input seed)
    seed_index = np.random.randint(0, vocab_size)
    print("Using seed index:", seed_index)
    
    # Generate text indices and decode them to characters
    generated_indices = model.sample(seed_index, num_chars)
    generated_text = "".join(chr(ix) for ix in generated_indices)
    print("Generated text:")
    print(generated_text)
```

## Conclusion

By rolling the same transformation across layers, we maintain an architecture with the same depth (number of iterations) as a multi‑layer network, but with a dramatically reduced number of parameters due to weight sharing. This approach can be beneficial in production settings where efficiency and consistency are paramount, while still achieving effects similar to a deep network.
```` ▋
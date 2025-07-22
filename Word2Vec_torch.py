import numpy as np
import pandas as pd
import ast
import torch
from collections import defaultdict

class Word2VecModel:
    def __init__(self, filename, embedding_dim=100, window_size=3, learning_rate=0.01, epochs=10):
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize class variables
        self.filename = filename
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Load the data
        self.df = pd.read_csv(self.filename, header=0)
        self.df['Berita'] = self.df['Berita'].apply(ast.literal_eval)
        
        # Flatten and create vocabulary
        self.flattened_tokens = [token for tokens in self.df['Berita'] for token in tokens]
        self.word_to_id = self.create_vocabulary(self.flattened_tokens)
        
        # Generate training data
        self.training_data = self.generate_training_data(self.df['Berita'], self.word_to_id)
        
        # Initialize weights
        self.w, self.v = self.initialize_weights(len(self.word_to_id))

        # Move weights to the appropriate device
        self.w = self.w.to(self.device)
        self.v = self.v.to(self.device)

    def create_vocabulary(self, tokens):
        """Create vocabulary from tokens."""
        word_count = defaultdict(int)
        for token in tokens:
            word_count[token] += 1
        word_to_id = {word: i for i, (word, _) in enumerate(word_count.items())}
        return word_to_id

    def generate_training_data(self, data, word_to_id):
        """Generate training data as context-y pairs."""
        training_data = []
        for tokens in data:
            for i, word in enumerate(tokens):
                start = max(0, i - self.window_size)
                end = min(len(tokens), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        training_data.append((word_to_id[word], word_to_id[tokens[j]]))
        return training_data

    def initialize_weights(self, vocab_size):
        """Initialize weights for the neural network."""
        # create 
        np.random.seed(seed=42) # for reproducibility
        
        w = torch.rand(vocab_size, self.embedding_dim, device=self.device) * np.sqrt(2 / (vocab_size + self.embedding_dim)) # Input weights
        v = torch.rand(self.embedding_dim, vocab_size, device=self.device) * np.sqrt(2 / (vocab_size + self.embedding_dim)) # Output weights
        return w, v

    def one_hot_encode(self, idx, vocab_size):
        """One-hot encode the input index and return a PyTorch tensor."""
        res = torch.zeros(vocab_size, device=self.device)  # Use torch.zeros
        res[idx] = 1
        return res

    def softmax(self, x):
        """Compute the softmax of a vector manually."""
        exp_x = torch.exp(x - torch.max(x))  # Subtract max(x) to improve stability
        return exp_x / torch.sum(exp_x)

    def forward_propagation(self, word, w, v):
        """Perform forward propagation."""
        # Input Layer
        X = self.one_hot_encode(word, len(w))
        
        # Hidden Layer
        A = torch.matmul(X, w)

        # Output Layer
        B = torch.matmul(A, v)
        Z = self.softmax(B)

        return A, B, Z

    def backward_propagation(self, word, context, A, Z, w, v):
        """Perform backward propagation."""
        y = self.one_hot_encode(context, len(w))

        dB = Z - y
        dv = torch.outer(A, dB)
        dA = torch.matmul(dB, v.T)
        dw = torch.outer(dA, self.one_hot_encode(word, len(w))).T

        # Update weights
        w -= self.learning_rate * dw
        v -= self.learning_rate * dv
        return w, v

    def train_word2vec(self):
        """Train the Word2Vec model using forward and backward propagation."""
        for epoch in range(self.epochs):
            total_loss = 0
            for word, context in self.training_data:
                # Forward pass
                A, B, Z = self.forward_propagation(word, self.w, self.v)
                
                # Compute loss (cross-entropy loss)
                y = self.one_hot_encode(context, len(self.w))
                L = -torch.sum(y * torch.log(Z + 1e-9))  # Adding 1e-9 for numerical stability
                total_loss += L
                
                # Backward pass
                self.w, self.v = self.backward_propagation(word, context, A, Z, self.w, self.v)

            # Print loss for the current epoch
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss.item():.4f}")

    def aggregate_vectors_per_document(self, data):
        """Aggregate word vectors for each document."""
        V = []
        for tokens in data:
            token_indices = [self.word_to_id[word] for word in tokens if word in self.word_to_id]
            doc_vector = torch.mean(self.w[token_indices], dim=0)
            V.append(doc_vector.cpu().numpy())  # Convert to numpy array for storage
        return np.array(V)

    def save_vectors_to_csv(self, output_file):
        """Save word2vec vectors to CSV."""
        V = self.aggregate_vectors_per_document(self.df['Berita'])
        self.df['Word2Vec Vector'] = V.tolist()
        self.df.to_csv(output_file, index=False)
        print(f"Word2Vec vectors saved to {output_file}")


if __name__ == "__main__":
    # Filepath to CSV file
    filename = "Preprocessing/Sample Dataset Preprocessing.csv"

    # Initialize the Word2Vec model
    model = Word2VecModel(filename)

    # Train the model
    model.train_word2vec()

    # Save the Word2Vec vectors to a CSV file
    output_file = 'Word2vec_Vector_100_epoch.csv'
    model.save_vectors_to_csv(output_file)

import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # create 
        np.random.seed(seed=42) # for reproducibility

        self.w = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(2 / (input_size + hidden_size))
        # print("Shape w", self.w.shape)
        self.v = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(2 / (hidden_size + output_size))
        # print("Shape v", self.v.shape)
        self.b = np.random.randn(self.hidden_size, 1) 
        # print("Shape b", self.b.shape)
        self.c = np.random.randn(self.output_size, 1) 
        # print("Shape c", self.c.shape)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z))
        return expZ / (expZ.sum(axis=0, keepdims=True) + 1e-10) 
    
    def deriv_relu(self, z):
        return np.where(z > 0, 1, 0)

    def one_hot_encode(self, Y):
        '''One hot encode the given input data.'''
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    
    def forward(self, X):
        P = np.dot(self.w, X) + self.b
        # print("P: ", P.shape)
        p_activation = self.relu(P)
        # print("p_activation: ", p_activation.shape)
        Q = np.dot(self.v, p_activation) + self.c
        # print("Q: ", Q.shape)
        q_activation = self.softmax(Q)
        # print("q_activation: ", q_activation.shape)
        return X, P, p_activation, Q, q_activation
    
    def backward(self, X, Y, P, p_activation, Q, q_activation):
        one_hot_Y = self.one_hot_encode(Y)
        m = X.shape[1]
        # print("one_hot_Y", one_hot_Y.shape)
        # print("X: ", X.shape)
        # print("m: ", m)

        # Gradient descent hidden layer 2, v, and ,c
        dQ = q_activation - one_hot_Y
        # print("dQ: ", dQ.shape)
        dv = (1/m) * np.dot(dQ, p_activation.T)
        # print("dv: ", dv.shape)
        dc = (1/m) * np.sum(dQ, 1)
        # print("dc: ", dc.shape)
        
        # Gradient descent hidden layer 1, w, and ,b
        dP = np.dot(self.v.T, dQ) * self.deriv_relu(P)
        # print("dP: ", dP.shape)
        dw = (1/m) * np.dot(dP, X.T)
        # print("dw: ", dw.shape)
        db = (1/m) * np.sum(dP, 1)
        # print("db: ", db.shape)
        return dw, db, dv, dc
    
    def update_weight_bias(self, dw, db, dv, dc, learning_rate):
        self.w -= learning_rate * dw
        self.b -= learning_rate * np.reshape(db, (self.hidden_size, 1))
        self.v -= learning_rate * dv
        self.c -= learning_rate * np.reshape(dc, (self.output_size, 1))
    
    def train_until_cost_doesnt_change(self, X, Y, learning_rate, X_val=None, Y_val=None):
        '''Train the neural network using the given input and output data until the cost doesn't change.'''
        history_cost = []
        history_acc = []
        val_history_cost = []
        val_history_acc = []

        cost = 0
        epoch = 0
        patience = 10
        while True:
            _, P, p_activation, Q, q_activation = self.forward(X)
            dw, db, dv, dc = self.backward(X, Y, P, p_activation, Q, q_activation)
            self.update_weight_bias(dw, db, dv, dc, learning_rate)
            new_cost = -np.mean(self.one_hot_encode(Y) * np.log(q_activation + 1e-8))
            Z = np.argmax(q_activation, 0)
            train_acc = np.mean(Z == Y)  # compute accuration

            history_cost.append(new_cost)
            history_acc.append(train_acc)
            
            # Validation Metrics
            if X_val is not None and Y_val is not None:
                _, _, _, _, q_val = self.forward(X_val)
                val_cost = -np.mean(self.one_hot_encode(Y_val) * np.log(q_val + 1e-8))
                y_val_pred = np.argmax(q_val, axis=0)
                val_acc = np.mean(y_val_pred == Y_val)
                val_history_cost.append(val_cost)
                val_history_acc.append(val_acc)
                print(f'Epoch {epoch + 1} - Train Loss: {new_cost:.6f}, Train Acc: {train_acc:.6f} | Val Loss: {val_cost:.6f}, Val Acc: {val_acc:.6f}')
            else:
                print(f'Epoch {epoch + 1} - Train Loss: {new_cost:.6f}, Train Acc: {train_acc:.6f}')
                            
            # early stopping with patience
            if abs(new_cost - cost) < 1e-6:
                patience -= 1
                if patience == 0:
                    print('Cost has not changed for 10 epochs, stopping training.')
                    break
            else:
                patience = 5
            cost = new_cost
            epoch += 1
        return history_cost, history_acc, val_history_cost, val_history_acc

    def print_info(self, npObject, title):
        print("---- ", title, " ----")
        print("Shape: ", npObject.shape)
        # print("Type: ", npObject.dtype)
        print("Size: ", npObject.size)
        print("Dimension: ", npObject.ndim)
        print("isNumpy: ", isinstance(npObject, np.ndarray))
        print("Data: ", npObject)
        print("\n")

    def predict(self, X, Y):
        '''Predict the output based on the given input data.'''
        _, _, _, _, q_activation = self.forward(X)
        y_pred = np.argmax(q_activation, axis=0)
        if Y.ndim == 2:
            Y = np.argmax(Y, axis=1)
        return y_pred, Y

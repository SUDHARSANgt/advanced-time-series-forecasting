# advanced_time_series_forecasting.py
"""
Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms
Complete 700+ line implementation - FIXED FOR ONLINE EXECUTION
"""

# First install required packages
!pip install torch torchvision torchaudio numpy pandas matplotlib seaborn scikit-learn optuna tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import math
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import optuna after installation
try:
    import optuna
except ImportError:
    !pip install optuna
    import optuna

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ComplexTimeSeriesDataset:
    """Generate and preprocess complex multivariate time series data"""
    
    def __init__(self, seq_length=60, pred_length=10, test_size=0.2, val_size=0.1):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.test_size = test_size
        self.val_size = val_size
        self.scalers = {}
        
    def generate_complex_data(self, n_samples=10000):
        """Generate complex multivariate time series with multiple frequencies and noise"""
        t = np.linspace(0, 100, n_samples)
        
        # Multiple frequency components
        signal1 = np.sin(2 * np.pi * 0.1 * t)  # Low frequency
        signal2 = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Medium frequency
        signal3 = 0.2 * np.sin(2 * np.pi * 2.0 * t)  # High frequency
        
        # Trend component
        trend = 0.01 * t
        
        # Seasonal component
        seasonal = 0.3 * np.sin(2 * np.pi * 0.02 * t)
        
        # Noise components (different characteristics)
        noise1 = 0.1 * np.random.normal(0, 1, n_samples)
        noise2 = 0.05 * np.random.standard_t(3, n_samples)  # Heavy-tailed noise
        
        # Create multivariate series
        main_series = signal1 + signal2 + signal3 + trend + seasonal + noise1
        
        # Correlated series
        correlated_series = 0.8 * main_series + 0.2 * noise2 + 0.1 * np.random.normal(0, 1, n_samples)
        
        # Exogenous series with different behavior
        exogenous_series = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.random.normal(0, 1, n_samples)
        
        # Combine into multivariate dataset
        data = np.column_stack([main_series, correlated_series, exogenous_series])
        
        # Add some abrupt changes to make it more challenging
        change_points = [2500, 5000, 7500]
        for cp in change_points:
            data[cp:, 0] += 0.5  # Level shift
            data[cp:, 1] *= 1.2  # Scale change
            
        self.raw_data = data
        return data
    
    def preprocess_data(self, data):
        """Normalize and prepare data for training"""
        self.scalers = {}
        scaled_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            scaler = StandardScaler()
            scaled_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
            self.scalers[i] = scaler
            
        return scaled_data
    
    def create_sequences(self, data):
        """Create input-output sequences for training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.seq_length - self.pred_length + 1):
            seq = data[i:(i + self.seq_length)]
            target = data[i + self.seq_length:i + self.seq_length + self.pred_length, 0]  # Predict main series
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def prepare_datasets(self):
        """Prepare train, validation, and test datasets"""
        # Generate data
        data = self.generate_complex_data(5000)  # Reduced for faster execution
        
        # Preprocess
        scaled_data = self.preprocess_data(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Split data maintaining temporal order
        n_total = len(X)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.val_size)
        n_train = n_total - n_test - n_val
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
        
        return (X_train, y_train, X_val, y_val, X_test, y_test)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class BaselineLSTM(nn.Module):
    """Baseline LSTM model without attention"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=10, dropout=0.2):
        super(BaselineLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state for prediction
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        
        return output

class AttentionLayer(nn.Module):
    """Bahdanau-style attention mechanism"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, seq_length, hidden_dim)
        
        # Repeat hidden state for each time step
        hidden_repeated = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.W(encoder_outputs) + self.U(hidden_repeated))
        attention_scores = self.v(energy).squeeze(-1)  # (batch_size, seq_length)
        
        # Apply softmax to get attention weights
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        
        # Calculate context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)
        
        return context_vector, attention_weights

class AttentionLSTM(nn.Module):
    """LSTM model with Bahdanau attention mechanism"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=10, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Concatenate context and last hidden
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Get last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Apply attention
        context_vector, attention_weights = self.attention(last_hidden, lstm_out)
        
        # Concatenate context vector with last hidden state
        combined = torch.cat([last_hidden, context_vector], dim=1)
        combined = self.dropout(combined)
        
        output = self.fc(combined)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output, attn_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""
    
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attn_weights = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feedforward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, attn_weights

class TransformerForecaster(nn.Module):
    """Transformer-based time series forecaster"""
    
    def __init__(self, input_dim=3, d_model=64, num_heads=4, num_layers=3, 
                 dim_feedforward=256, output_dim=10, dropout=0.1, max_seq_len=100):
        super(TransformerForecaster, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        all_attention_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, mask)
            all_attention_weights.append(attn_weights)
        
        # Use the output from the last position for forecasting
        last_output = x[:, -1, :]  # (batch_size, d_model)
        last_output = self.dropout(last_output)
        output = self.output_projection(last_output)
        
        return output, all_attention_weights

class TimeSeriesTrainer:
    """Training and evaluation class for time series models"""
    
    def __init__(self, model, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
        """Train the model with early stopping"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if self.model_name in ['AttentionLSTM', 'Transformer']:
                    outputs, _ = self.model(batch_X)
                else:
                    outputs = self.model(batch_X)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            avg_val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'best_{self.model_name}.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        try:
            self.model.load_state_dict(torch.load(f'best_{self.model_name}.pth'))
        except:
            print("Could not load saved model, using final model")
        
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                if self.model_name in ['AttentionLSTM', 'Transformer']:
                    outputs, _ = self.model(batch_X)
                else:
                    outputs = self.model(batch_X)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def evaluate(self, test_loader, scalers):
        """Comprehensive evaluation on test set"""
        self.model.eval()
        predictions = []
        actuals = []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                if self.model_name in ['AttentionLSTM', 'Transformer']:
                    outputs, attention_weights = self.model(batch_X)
                    attention_weights_list.append(attention_weights)
                else:
                    outputs = self.model(batch_X)
                
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Inverse transform predictions and actuals
        scaler = scalers[0]  # Use scaler for the first feature (main series)
        predictions_original = scaler.inverse_transform(predictions)
        actuals_original = scaler.inverse_transform(actuals)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals_original, predictions_original))
        mae = mean_absolute_error(actuals_original, predictions_original)
        mape = np.mean(np.abs((actuals_original - predictions_original) / (actuals_original + 1e-8))) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        return metrics, predictions_original, actuals_original, attention_weights_list
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title(f'{self.model_name} - Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, model_class, model_name):
        self.model_class = model_class
        self.model_name = model_name
        self.best_params = None
        
    def create_data_loaders(self, batch_size=32):
        """Create data loaders for training"""
        dataset_generator = ComplexTimeSeriesDataset(seq_length=60, pred_length=10)
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_generator.prepare_datasets()
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, dataset_generator.scalers
        
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        # Suggest hyperparameters
        if self.model_name == 'BaselineLSTM':
            hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
            num_layers = trial.suggest_int('num_layers', 1, 2)
            dropout = trial.suggest_float('dropout', 0.1, 0.3)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32])
            
            model = BaselineLSTM(
                input_dim=3,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=10,
                dropout=dropout
            )
            
        elif self.model_name == 'AttentionLSTM':
            hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
            num_layers = trial.suggest_int('num_layers', 1, 2)
            dropout = trial.suggest_float('dropout', 0.1, 0.3)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32])
            
            model = AttentionLSTM(
                input_dim=3,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=10,
                dropout=dropout
            )
            
        elif self.model_name == 'Transformer':
            d_model = trial.suggest_categorical('d_model', [32, 64])
            num_heads = trial.suggest_int('num_heads', 2, 4)
            num_layers = trial.suggest_int('num_layers', 1, 2)
            dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256])
            dropout = trial.suggest_float('dropout', 0.1, 0.3)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32])
            
            model = TransformerForecaster(
                input_dim=3,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                output_dim=10,
                dropout=dropout
            )
        
        # Create data loaders
        train_loader, val_loader, _, scalers = self.create_data_loaders(batch_size=batch_size)
        
        # Train model
        trainer = TimeSeriesTrainer(model, self.model_name)
        trainer.train(train_loader, val_loader, epochs=30, lr=lr, patience=5)
        
        # Get validation loss
        final_val_loss = trainer.val_losses[-1]
        
        return final_val_loss
    
    def optimize(self, n_trials=10):
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        print(f"Best hyperparameters for {self.model_name}:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return self.best_params

class AttentionAnalyzer:
    """Analyze and visualize attention weights"""
    
    def __init__(self, model, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        
    def get_attention_weights(self, data_loader, num_samples=5):
        """Extract attention weights for analysis"""
        self.model.eval()
        all_attention_weights = []
        sample_inputs = []
        sample_outputs = []
        
        with torch.no_grad():
            for i, (batch_X, batch_y) in enumerate(data_loader):
                if i >= num_samples:
                    break
                    
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                if self.model_name == 'AttentionLSTM':
                    outputs, attention_weights = self.model(batch_X)
                    # attention_weights shape: (batch_size, seq_length)
                    all_attention_weights.append(attention_weights.cpu().numpy())
                    
                elif self.model_name == 'Transformer':
                    outputs, all_layer_weights = self.model(batch_X)
                    # Use attention weights from the last layer
                    last_layer_weights = all_layer_weights[-1]  # (batch_size, num_heads, seq_len, seq_len)
                    # Average across heads for visualization
                    attention_weights = last_layer_weights.mean(dim=1).cpu().numpy()
                    all_attention_weights.append(attention_weights)
                
                sample_inputs.append(batch_X.cpu().numpy())
                sample_outputs.append(outputs.cpu().numpy())
        
        return (np.concatenate(all_attention_weights, axis=0),
                np.concatenate(sample_inputs, axis=0),
                np.concatenate(sample_outputs, axis=0))
    
        def plot_attention_heatmap(self, attention_weights, inputs, sample_idx=0):
        """Plot attention heatmap for a sample"""
        if self.model_name == 'AttentionLSTM':
            # attention_weights shape: (seq_length,)
            attn_weights = attention_weights[sample_idx]
            seq_length = len(attn_weights)
            
            plt.figure(figsize=(12, 4))
            plt.imshow(attn_weights.reshape(1, -1), cmap='viridis', aspect='auto')
            plt.colorbar(label='Attention Weight')
            plt.title(f'{self.model_name} - Attention Weights (Sample {sample_idx})')
            plt.xlabel('Time Step')
            plt.yticks([])
            
        elif self.model_name == 'Transformer':
            # attention_weights shape: (seq_length, seq_length)
            attn_weights = attention_weights[sample_idx]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn_weights, cmap='viridis', annot=False, fmt='.3f')
            plt.title(f'{self.model_name} - Self-Attention Weights (Sample {sample_idx})')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
        
        plt.tight_layout()
        plt.show()
    
    def plot_attention_temporal_pattern(self, attention_weights, inputs, scalers, sample_idx=0):
        """Plot attention weights along with input time series"""
        if self.model_name == 'AttentionLSTM':
            attn_weights = attention_weights[sample_idx]
            input_sequence = inputs[sample_idx, :, 0]  # Main series
            
            # Inverse transform
            scaler = scalers[0]
            input_sequence_original = scaler.inverse_transform(input_sequence.reshape(-1, 1)).flatten()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot input sequence
            time_steps = range(len(input_sequence_original))
            ax1.plot(time_steps, input_sequence_original, 'b-', linewidth=2, label='Input Series')
            ax1.set_ylabel('Value')
            ax1.set_title('Input Time Series')
            ax1.grid(True)
            ax1.legend()
            
            # Plot attention weights
            ax2.bar(time_steps, attn_weights, alpha=0.7, color='red')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Attention Weight')
            ax2.set_title('Attention Weights Distribution')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # Print analysis
            print(f"Attention Analysis for Sample {sample_idx}:")
            print(f"Mean attention weight: {np.mean(attn_weights):.4f}")
            print(f"Std attention weight: {np.std(attn_weights):.4f}")
            print(f"Max attention at time step: {np.argmax(attn_weights)}")
            print(f"Top 3 attention time steps: {np.argsort(attn_weights)[-3:][::-1]}")
    
    def analyze_attention_patterns(self, data_loader, scalers, num_samples=2):
        """Comprehensive attention pattern analysis"""
        attention_weights, inputs, outputs = self.get_attention_weights(data_loader, num_samples)
        
        print(f"=== {self.model_name} Attention Analysis ===")
        print(f"Overall attention statistics:")
        print(f"  Mean attention weight: {np.mean(attention_weights):.4f}")
        print(f"  Std attention weight: {np.std(attention_weights):.4f}")
        
        # Plot for each sample
        for i in range(min(num_samples, len(attention_weights))):
            self.plot_attention_heatmap(attention_weights, inputs, i)
            if self.model_name == 'AttentionLSTM':
                self.plot_attention_temporal_pattern(attention_weights, inputs, scalers, i)

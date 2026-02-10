import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import re
 
# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameter configuration
HYPERPARAMETERS = {
    # Data configuration
    'TRAIN_START': 4096,
    'TRAIN_END': 8192,
    'EPOCHS': 30,  # 设置为标准训练轮数
    'BATCH_SIZE': 32,
    'TEST_RATIO': 0.2,  # Test set accounts for 1/5 of training set
    'CONTINUE_TRAINING': True,
    'DEBUG_MODE': False,
    
    # Top-N accuracy calculation configuration
    'TOP_N_MAX': 10,  # Calculate Top-1 to Top-10 accuracy
    
    # Early Stopping configuration
    'EARLY_STOPPING_PATIENCE': 20,
    'EARLY_STOPPING_MIN_DELTA': 1e-4,
    'EARLY_STOPPING_MONITOR': 'test_loss',
    
    # Model configuration
    'N_QUBITS': 12,
    'N_LAYERS': 3,
    'INPUT_DIM': 48,
    'TOTAL_FEATURES': 256,  # Total features
    
    # MLR (classic MLP post-processing) structure configuration
    'MLR_HIDDEN_DIM': 64,  # MLR hidden layer dimension
    'MLR_ACTIVATION': 'ReLU',  # MLR activation function
    
    # Training configuration
    'LEARNING_RATE': 0.001,
    'SHUFFLE_TRAIN': False,
    
    # Other configuration
    'DATA_PATH': '/Users/luxian/DataSpace/beam_pre/sls_beam_data_spatial_domain_vivo.mat',
    'OUTPUT_DIR': 'pqc_reup_v1_output'
}

# Calculate derived parameters (TRAIN_END is now fixed, not calculated)
HYPERPARAMETERS['TRAIN_SAMPLES'] = HYPERPARAMETERS['TRAIN_END'] - HYPERPARAMETERS['TRAIN_START']
HYPERPARAMETERS['OUTPUT_DIM'] = HYPERPARAMETERS['TOTAL_FEATURES'] - HYPERPARAMETERS['INPUT_DIM']
HYPERPARAMETERS['TEST_SAMPLES'] = int(HYPERPARAMETERS['TRAIN_SAMPLES'] * HYPERPARAMETERS['TEST_RATIO'])
# HYPERPARAMETERS['TEST_START'] = HYPERPARAMETERS['TRAIN_END']
HYPERPARAMETERS['TEST_START'] = 280000
HYPERPARAMETERS['TEST_END'] = HYPERPARAMETERS['TEST_START'] + HYPERPARAMETERS['TEST_SAMPLES']

print(f"Dual Top-N Methods Test Configuration:")
print(f"Input beam count: {HYPERPARAMETERS['INPUT_DIM']}")
print(f"Output beam count: {HYPERPARAMETERS['OUTPUT_DIM']}")
print(f"Total beam count: {HYPERPARAMETERS['TOTAL_FEATURES']}")
print(f"Training samples: {HYPERPARAMETERS['TRAIN_SAMPLES']}")
print(f"Testing samples: {HYPERPARAMETERS['TEST_SAMPLES']}")
print(f"Calculate both Top-N methods: A(including input beams) vs B(excluding input beams)")
print(f"Early Stopping configuration:")
print(f"  Patience: {HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']} epochs")
print(f"  Min Delta: {HYPERPARAMETERS['EARLY_STOPPING_MIN_DELTA']}")
print(f"  Monitor: {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']}")
print(f"Configuration verification:")
print(f"  TRAIN_END is directly specified as {HYPERPARAMETERS['TRAIN_END']} samples")
print(f"  TRAIN_START: {HYPERPARAMETERS['TRAIN_START']}")
print(f"  TRAIN_SAMPLES: {HYPERPARAMETERS['TRAIN_SAMPLES']}")
print(f"  TEST_START: {HYPERPARAMETERS['TEST_START']}")
print(f"  TEST_END: {HYPERPARAMETERS['TEST_END']}")
print(f"MLR structure: {HYPERPARAMETERS['N_QUBITS']} → {HYPERPARAMETERS['MLR_HIDDEN_DIM']} → {HYPERPARAMETERS['OUTPUT_DIM']}")

class QuantumDataReuploadModel(nn.Module):
    """Quantum beam prediction model based on data re-upload technology"""
    
    def __init__(self, n_qubits=HYPERPARAMETERS['N_QUBITS'], n_layers=HYPERPARAMETERS['N_LAYERS'], 
                 input_dim=HYPERPARAMETERS['INPUT_DIM'], output_dim=HYPERPARAMETERS['OUTPUT_DIM']):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.chunk_size = 4  # Each chunk has 4 features
        self.n_chunks = input_dim // self.chunk_size  # 12 chunks
        
        # Data re-upload parameters: map 4D to 3D weights and biases
        self.reupload_weights = nn.Parameter(torch.randn(self.n_chunks, 3, self.chunk_size) * 0.1)
        self.reupload_bias = nn.Parameter(torch.randn(self.n_chunks, 3) * 0.1)
        
        # Quantum device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")
        
        # MLR (Multi-Layer Regression) post-processing layer
        # Structure: quantum output(n_qubits) -> hidden layer(MLR_HIDDEN_DIM) -> output layer(output_dim)
        self.mlr = nn.Sequential(
            nn.Linear(n_qubits, HYPERPARAMETERS['MLR_HIDDEN_DIM']),
            nn.ReLU(),
            nn.Linear(HYPERPARAMETERS['MLR_HIDDEN_DIM'], output_dim)
        )
        
    def quantum_circuit(self, params):
        """Quantum circuit definition"""
        # Strong entanglement layer
        for layer in range(self.n_layers):
            # Data re-upload
            for i in range(self.n_chunks):
                chunk_params = params[i]
                # U3 gate encoding
                qml.U3(chunk_params[0], chunk_params[1], chunk_params[2], wires=i)
            
            # Strong entanglement operation
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                
        # Measure Pauli-Z expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """Forward propagation"""
        batch_size = x.shape[0]
        quantum_outputs = []
        
        # Process each sample
        for i in range(batch_size):
            sample = x[i]
            
            # Split input into 12 chunks of 4D
            chunks = sample.view(self.n_chunks, self.chunk_size)
            
            # Data re-upload: map each chunk to 3D parameters
            processed_chunks = []
            for j in range(self.n_chunks):
                chunk = chunks[j]
                # Linear transformation: W * x + b -> 3D
                transformed = torch.matmul(self.reupload_weights[j], chunk) + self.reupload_bias[j]
                processed_chunks.append(transformed)
            
            # Quantum circuit processing
            quantum_params = torch.stack(processed_chunks)
            q_out = self.qnode(quantum_params)
            quantum_outputs.append(torch.stack(q_out))
        
        # Stack batch results
        quantum_output = torch.stack(quantum_outputs)
        
        # Ensure data type is float32
        quantum_output = quantum_output.float()
        
        # MLR post-processing
        final_output = self.mlr(quantum_output)
        
        return final_output

def load_and_preprocess_data(filepath, train_start=HYPERPARAMETERS['TRAIN_START'], 
                           train_end=HYPERPARAMETERS['TRAIN_END']):
    """Load and preprocess data, divide into training and test sets by specified range"""
    # Automatically calculate test set range
    train_samples = train_end - train_start
    test_samples = int(train_samples * HYPERPARAMETERS['TEST_RATIO'])
    test_start = train_end
    test_end = test_start + test_samples
    
    print(f"Data division: Training set[{train_start}:{train_end}] ({train_samples} samples), "
          f"Test set[{test_start}:{test_end}] ({test_samples} samples)")
    
    # Use h5py to load MATLAB v7.3 file
    try:
        with h5py.File(filepath, 'r') as f:
            # Try common data key names
            data_keys = ['beam_data', 'data', 'X', 'features', 'rsrp']
            data = None
            
            for key in data_keys:
                if key in f:
                    data = np.array(f[key]).T  # MATLAB storage is usually transposed
                    break
            
            # If no common key names are found, try the first dataset
            if data is None:
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data = np.array(f[key]).T
                        break
            
            if data is None:
                raise ValueError("No valid dataset found in HDF5 file")
                
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Ensure data is 2D
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    
    # Check if index range is valid (now in sample dimension)
    total_samples = data.shape[1]  # Sample count in the second dimension
    if train_end > total_samples or test_end > total_samples:
        # Adjust to valid range
        train_end = min(train_end, total_samples)
        test_end = min(test_end, total_samples)
        print(f"Adjusted range: Training set[{train_start}:{train_end}], Test set[{test_start}:{test_end}]")
    
    # Select samples by specified range (in the second dimension)
    train_indices = np.arange(train_start, train_end)
    test_indices = np.arange(test_start, test_end)
    
    # Select samples
    X_train_full = data[:, train_indices].T  # Transpose to (sample count, feature count)
    X_test_full = data[:, test_indices].T
    
    # Equidistantly select 48 features (in the first dimension)
    n_features = data.shape[0]  # Feature count
    input_indices = np.linspace(0, n_features-1, HYPERPARAMETERS['INPUT_DIM'], dtype=int)
    
    X_train_input = X_train_full[:, input_indices]
    X_test_input = X_test_full[:, input_indices]
    
    # Output is all remaining features
    all_indices = set(range(n_features))
    output_indices = list(all_indices - set(input_indices))
    X_train_output = X_train_full[:, output_indices]
    X_test_output = X_test_full[:, output_indices]
    
    # Data normalization
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_input_norm = scaler_X.fit_transform(X_train_input)
    X_test_input_norm = scaler_X.transform(X_test_input)
    
    X_train_output_norm = scaler_y.fit_transform(X_train_output)
    X_test_output_norm = scaler_y.transform(X_test_output)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_input_norm)
    y_train_tensor = torch.FloatTensor(X_train_output_norm)
    X_test_tensor = torch.FloatTensor(X_test_input_norm)
    y_test_tensor = torch.FloatTensor(X_test_output_norm)
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_X, scaler_y, input_indices, output_indices, train_indices, test_indices)

def load_historical_training_data(output_dir, start_epoch):
    """加载历史训练数据"""
    if start_epoch <= 0:
        return [], [], []
    
    # 查找最接近start_epoch的历史训练数据文件
    training_data_files = [f for f in os.listdir(output_dir) if f.startswith('training_data_epoch_') and f.endswith('.json')]
    
    if not training_data_files:
        return [], [], []
    
    # 提取epoch编号并找到最合适的文件
    epochs = []
    for f in training_data_files:
        try:
            epoch = int(f.split('_')[3].split('.')[0])
            if epoch <= start_epoch:
                epochs.append((epoch, f))
        except:
            continue
    
    if not epochs:
        return [], [], []
    
    # 选择最接近且不超过start_epoch的文件
    epochs.sort(reverse=True)  # 从大到小排序
    best_epoch, best_file = epochs[0]
    
    try:
        with open(os.path.join(output_dir, best_file), 'r') as f:
            historical_data = json.load(f)
        
        historical_train_losses = historical_data.get('train_losses', [])
        historical_test_losses = historical_data.get('test_losses', [])
        historical_mae_scores = historical_data.get('mae_scores', [])
        
        print(f"成功加载历史训练数据: {best_file} (Epoch {best_epoch}, {len(historical_train_losses)} 个数据点)")
        return historical_train_losses, historical_test_losses, historical_mae_scores
    except Exception as e:
        print(f"加载历史训练数据失败: {e}")
        return [], [], []

def merge_training_data(historical_train, historical_test, historical_mae, 
                       new_train, new_test, new_mae):
    """合并历史训练数据和新训练数据"""
    # 合并数据
    merged_train = historical_train + new_train
    merged_test = historical_test + new_test
    merged_mae = historical_mae + new_mae
    
    print(f"训练数据合并完成:")
    print(f"  历史数据点: {len(historical_train)}")
    print(f"  新增数据点: {len(new_train)}")
    print(f"  合并后总计: {len(merged_train)}")
    
    return merged_train, merged_test, merged_mae

def train_model(model, train_loader, test_loader, 
                epochs=HYPERPARAMETERS['EPOCHS'], lr=HYPERPARAMETERS['LEARNING_RATE'],
                start_epoch=0):
    """Train model, including Early Stopping mechanism and debug mode support"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 如果是继续训练，加载历史训练数据
    if start_epoch > 0:
        historical_train, historical_test, historical_mae = load_historical_training_data(
            HYPERPARAMETERS['OUTPUT_DIR'], start_epoch
        )
    else:
        historical_train, historical_test, historical_mae = [], [], []
    
    train_losses = []
    test_losses = []
    mae_scores = []
    
    # Early Stopping related variables
    best_test_loss = float('inf')
    early_stopping_counter = 0
    early_stop_triggered = False
    
    print("开始训练模型...")
    for epoch in range(start_epoch, start_epoch + epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_batch_count = 0
        
        # Debug mode: limit number of batches per epoch
        debug_batch_limit = 3  # Limit to first 3 batches in debug mode
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Check debug mode limit
            if HYPERPARAMETERS['DEBUG_MODE'] and batch_idx >= debug_batch_limit:
                break
                
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batch_count += 1
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / train_batch_count if train_batch_count > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        # Testing phase
        model.eval()
        total_test_loss = 0.0
        total_mae = 0.0
        test_batch_count = 0
        
        with torch.no_grad():
            # Debug mode: limit number of batches during testing
            debug_batch_limit = 3  # Limit to first 3 batches in debug mode
            
            for batch_idx, (data, target) in enumerate(test_loader):
                # Check debug mode limit
                if HYPERPARAMETERS['DEBUG_MODE'] and batch_idx >= debug_batch_limit:
                    break
                    
                output = model(data)
                
                # Calculate MSE loss
                mse_loss = criterion(output, target)
                total_test_loss += mse_loss.item()
                
                # Calculate MAE using PyTorch for consistency
                mae = torch.mean(torch.abs(output - target))
                total_mae += mae.item()
                
                test_batch_count += 1
        
        # Calculate average test metrics
        avg_test_loss = total_test_loss / test_batch_count if test_batch_count > 0 else 0.0
        avg_mae = total_mae / test_batch_count if test_batch_count > 0 else 0.0
        
        test_losses.append(avg_test_loss)
        mae_scores.append(avg_mae)
        
        # Early Stopping check - decide whether to stop based on monitored metric
        if avg_test_loss < best_test_loss - HYPERPARAMETERS['EARLY_STOPPING_MIN_DELTA']:
            best_test_loss = avg_test_loss
            early_stopping_counter = 0
            # Save best model
            best_model_path = os.path.join(HYPERPARAMETERS['OUTPUT_DIR'], f'best_model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stopping_counter += 1
            
        # Check if early stopping is triggered
        if early_stopping_counter >= HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']:
            print(f"Epoch {epoch}: Early stopping triggered! {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']} hasn't improved for {HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']} epochs.")
            early_stop_triggered = True
            break
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            status = " (Early Stop)" if early_stop_triggered else ""
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, MAE: {avg_mae:.6f}{status}')
    
    # 合并历史数据和新训练数据
    if historical_train:
        train_losses, test_losses, mae_scores = merge_training_data(
            historical_train, historical_test, historical_mae,
            train_losses, test_losses, mae_scores
        )
    
    # Report Early Stopping status after training is complete
    final_epoch_count = len(train_losses)
    if early_stop_triggered:
        print(f"Training stopped early, total training rounds: {final_epoch_count}")
        print(f"Best {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']}: {best_test_loss:.6f}")
    else:
        print(f"Training completed {final_epoch_count} rounds, Early Stopping not triggered")
    
    return train_losses, test_losses, mae_scores

def evaluate_model(model, test_loader, scaler_y):
    """Evaluate model performance on test set"""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions.extend(output.numpy())
            targets.extend(target.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Inverse transform predictions and targets
    predictions_original = scaler_y.inverse_transform(predictions)
    targets_original = scaler_y.inverse_transform(targets)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(targets_original, predictions_original)
    mae = mean_absolute_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = 1 - (mse / np.var(targets_original))
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    return predictions_original, targets_original, metrics

def find_latest_model(output_dir):
    """查找最新的模型参数文件"""
    if not os.path.exists(output_dir):
        return None, 0
    
    model_files = [f for f in os.listdir(output_dir) if f.startswith('model_params_epoch_') and f.endswith('.pth')]
    
    if not model_files:
        return None, 0
    
    # 提取epoch编号
    epochs = []
    for f in model_files:
        try:
            epoch = int(f.split('_')[3].split('.')[0])
            epochs.append(epoch)
        except:
            continue
    
    if not epochs:
        return None, 0
    
    latest_epoch = max(epochs)
    latest_model_path = os.path.join(output_dir, f'model_params_epoch_{latest_epoch}.pth')
    
    return latest_model_path, latest_epoch

def load_latest_model(model, output_dir):
    """加载最新的模型参数"""
    model_path, epoch = find_latest_model(output_dir)
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"成功加载模型参数: {model_path} (Epoch {epoch})")
            return epoch
        except Exception as e:
            print(f"加载模型失败: {e}")
            return 0
    else:
        print("未找到可加载的模型文件，从头开始训练")
        return 0

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = f"{script_name}_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Data file path
    data_path = HYPERPARAMETERS['DATA_PATH']
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file does not exist: {data_path}")
        return
    
    # Load and preprocess data - divide by specified range
    (X_train, y_train, X_test, y_test, scaler_X, scaler_y, 
     input_indices, output_indices, train_indices, test_indices) = load_and_preprocess_data(
        data_path, 
        train_start=HYPERPARAMETERS['TRAIN_START'], 
        train_end=HYPERPARAMETERS['TRAIN_END']
    )
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=HYPERPARAMETERS['BATCH_SIZE'], 
                                             shuffle=HYPERPARAMETERS['SHUFFLE_TRAIN'])
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=HYPERPARAMETERS['BATCH_SIZE'], 
                                            shuffle=False)
    
    # Create model
    model = QuantumDataReuploadModel(
        n_qubits=HYPERPARAMETERS['N_QUBITS'], 
        n_layers=HYPERPARAMETERS['N_LAYERS'], 
        input_dim=HYPERPARAMETERS['INPUT_DIM'], 
        output_dim=HYPERPARAMETERS['OUTPUT_DIM']
    )
    
    # 处理是否继续训练的逻辑
    start_epoch = 0
    if HYPERPARAMETERS['CONTINUE_TRAINING']:
        # 尝试加载最新的模型参数以继续训练
        start_epoch = load_latest_model(model, output_dir)
    
    # 计算实际训练轮数
    actual_epochs = HYPERPARAMETERS['EPOCHS'] + start_epoch
    
    print(f"开始训练: {len(X_train)}训练样本, {len(X_test)}测试样本")
    print(f"输入维度: {X_train.shape[1]}, 输出维度: {y_train.shape[1]}")
    print(f"MLR结构: {HYPERPARAMETERS['N_QUBITS']} → {HYPERPARAMETERS['MLR_HIDDEN_DIM']} → {HYPERPARAMETERS['OUTPUT_DIM']}")
    print(f"训练样本数: {HYPERPARAMETERS['TRAIN_SAMPLES']}")
    print(f"输入波束数量: {len(input_indices)}, 输出波束数量: {len(output_indices)}")
    if start_epoch > 0 and HYPERPARAMETERS['CONTINUE_TRAINING']:
        print(f"继续训练: 从Epoch {start_epoch} 开始，训练至Epoch {actual_epochs-1}")
    elif start_epoch == 0 or not HYPERPARAMETERS['CONTINUE_TRAINING']:
        print(f"全新训练: 共 {HYPERPARAMETERS['EPOCHS']} 个epochs")
    print(f"起始Epoch: {start_epoch}")
    
    # 训练模型
    print("开始训练模型...")
    train_losses, test_losses, mae_scores = train_model(model, train_loader, test_loader, start_epoch=start_epoch)
    
    # 保存训练过程数据（使用合并后的完整数据）
    training_data = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'mae_scores': mae_scores,
        'epochs': len(train_losses),
        'training_info': {
            'start_epoch': start_epoch,
            'final_epoch': len(train_losses),
            'historical_epochs': len(train_losses) - HYPERPARAMETERS['EPOCHS'] if start_epoch > 0 else 0,
            'new_epochs': HYPERPARAMETERS['EPOCHS']
        }
    }
    
    # 保存模型参数（符合项目规范命名）
    final_epoch = len(train_losses)  # 使用合并后的总epoch数
    model_save_path = os.path.join(output_dir, f'model_params_epoch_{final_epoch}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"模型参数已保存到: {model_save_path}")
    
    # 保存完整的训练过程数据（包含历史+新数据）
    training_data_path = os.path.join(output_dir, f'training_data_epoch_{final_epoch}.json')
    with open(training_data_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"完整训练过程数据已保存到: {training_data_path}")
    
    # 同时保存当前运行的训练数据（仅本次训练的数据点）
    current_training_data = {
        'train_losses': train_losses[-HYPERPARAMETERS['EPOCHS']:] if start_epoch > 0 else train_losses,
        'test_losses': test_losses[-HYPERPARAMETERS['EPOCHS']:] if start_epoch > 0 else test_losses,
        'mae_scores': mae_scores[-HYPERPARAMETERS['EPOCHS']:] if start_epoch > 0 else mae_scores,
        'epochs': HYPERPARAMETERS['EPOCHS'],
        'training_info': {
            'start_epoch': start_epoch,
            'final_epoch': HYPERPARAMETERS['EPOCHS'],
            'is_current_run_only': True
        }
    }
    
    current_data_path = os.path.join(output_dir, f'training_data_current_run_epoch_{final_epoch}.json')
    with open(current_data_path, 'w') as f:
        json.dump(current_training_data, f, indent=2)
    print(f"当前运行训练数据已保存到: {current_data_path}")
    
    # 评估模型
    print("评估模型性能...")
    predictions_original, targets_original, metrics = evaluate_model(
        model, test_loader, scaler_y
    )
    
    # 保存完整的评估结果（包含真实指标）
    results = {
        'metrics': metrics,
        'predictions': predictions_original.tolist(),
        'targets': targets_original.tolist(),
        'training_info': {
            'start_epoch': start_epoch,
            'final_epoch': final_epoch,
            'total_epochs_trained': len(train_losses)
        }
    }
    
    results_path = os.path.join(output_dir, f'evaluation_results_epoch_{final_epoch}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"评估结果已保存到: {results_path}")
    
    # 调用分析模块生成图像
    print("调用分析模块生成结果图像...")
    try:
        import pqc_reup_analyze
        pqc_reup_analyze.analyze_results(
            epoch_num=len(train_losses),
            predictions=predictions_original,
            targets=targets_original,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"图像生成失败: {e}")
        print("继续执行其他操作...")
    
    # 保存评估指标（使用真实计算的指标，而非零值）
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"评估指标已保存到: {metrics_path}")
    
    print(f"\n训练完成!")
    print(f"输出文件保存在: {output_dir}")
    if start_epoch > 0:
        print(f"累计训练: {start_epoch} + {len(train_losses)} = {final_epoch} epochs")
        print(f"本次运行训练了 {HYPERPARAMETERS['EPOCHS']} 个epochs")
    else:
        print(f"总共训练: {len(train_losses)} epochs")

if __name__ == "__main__":
    main()
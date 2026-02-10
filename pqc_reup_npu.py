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

# =============================================================================
# NPU支持 - 导入torch_npu（必须在torch之后）
# =============================================================================
try:
    import torch_npu
    NPU_AVAILABLE = hasattr(torch, 'npu') and torch.npu.is_available()
    if NPU_AVAILABLE:
        print(f"✓ 检测到NPU设备: {torch.npu.device_count()} 个")
    else:
        print("ℹ️  NPU不可用，将使用CPU/CUDA")
except ImportError:
    NPU_AVAILABLE = False
    print("ℹ️  torch_npu未安装，NPU支持不可用")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameter configuration
HYPERPARAMETERS = {
    # Data configuration
    'TRAIN_START': 4096,
    'TRAIN_END': 8192,
    'EPOCHS': 30,
    'BATCH_SIZE': 32,
    'TEST_RATIO': 0.2,
    'CONTINUE_TRAINING': True,
    'DEBUG_MODE': False,
    
    # Top-N accuracy calculation configuration
    'TOP_N_MAX': 10,
    
    # Early Stopping configuration
    'EARLY_STOPPING_PATIENCE': 20,
    'EARLY_STOPPING_MIN_DELTA': 1e-4,
    'EARLY_STOPPING_MONITOR': 'test_loss',
    
    # Model configuration
    'N_QUBITS': 12,
    'N_LAYERS': 3,
    'INPUT_DIM': 48,
    'TOTAL_FEATURES': 256,
    
    # MLR structure configuration
    'MLR_HIDDEN_DIM': 64,
    'MLR_ACTIVATION': 'ReLU',
    
    # Training configuration
    'LEARNING_RATE': 0.001,
    'SHUFFLE_TRAIN': False,
    
    # Other configuration
    'DATA_PATH': '/Users/luxian/DataSpace/beam_pre/sls_beam_data_spatial_domain_vivo.mat',
    'OUTPUT_DIR': 'pqc_reup_npu_output'
}

# Calculate derived parameters
HYPERPARAMETERS['TRAIN_SAMPLES'] = HYPERPARAMETERS['TRAIN_END'] - HYPERPARAMETERS['TRAIN_START']
HYPERPARAMETERS['OUTPUT_DIM'] = HYPERPARAMETERS['TOTAL_FEATURES'] - HYPERPARAMETERS['INPUT_DIM']
HYPERPARAMETERS['TEST_SAMPLES'] = int(HYPERPARAMETERS['TRAIN_SAMPLES'] * HYPERPARAMETERS['TEST_RATIO'])
HYPERPARAMETERS['TEST_START'] = 280000
HYPERPARAMETERS['TEST_END'] = HYPERPARAMETERS['TEST_START'] + HYPERPARAMETERS['TEST_SAMPLES']

def get_device(device_str: str = 'auto') -> torch.device:
    """
    获取最优计算设备
    
    Args:
        device_str: 'auto', 'cpu', 'cuda', 'npu', 或具体设备如 'npu:0'
    
    Returns:
        torch.device对象
    """
    if device_str == 'auto':
        # NPU优先级最高
        if NPU_AVAILABLE:
            try:
                # 验证NPU实际可用性
                torch.npu.set_device(0)
                test_tensor = torch.zeros(1).npu()
                del test_tensor
                print("✓ NPU设备检测并验证成功")
                return torch.device('npu:0')
            except Exception as e:
                print(f"ℹ️  NPU初始化失败: {e}")
                print("_FALLBACK_ 到CUDA/CPU")
        
        # 回退到CUDA
        if torch.cuda.is_available():
            print("✓ 使用CUDA设备")
            return torch.device('cuda:0')
        
        # 最后使用CPU
        print("ℹ️  使用CPU设备")
        return torch.device('cpu')
    
    elif device_str == 'npu':
        if NPU_AVAILABLE:
            return torch.device('npu:0')
        else:
            raise RuntimeError("请求NPU但不可用")
    
    elif device_str == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            raise RuntimeError("请求CUDA但不可用")
    
    return torch.device(device_str)

class QuantumDataReuploadModel(nn.Module):
    """基于数据重上传技术的量子波束预测模型（NPU优化版本）"""
    
    def __init__(self, n_qubits=HYPERPARAMETERS['N_QUBITS'], n_layers=HYPERPARAMETERS['N_LAYERS'], 
                 input_dim=HYPERPARAMETERS['INPUT_DIM'], output_dim=HYPERPARAMETERS['OUTPUT_DIM']):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.chunk_size = 4
        self.n_chunks = input_dim // self.chunk_size
        
        # 数据重上传参数：将4D映射到3D的权重和偏置
        self.reupload_weights = nn.Parameter(torch.randn(self.n_chunks, 3, self.chunk_size) * 0.1)
        self.reupload_bias = nn.Parameter(torch.randn(self.n_chunks, 3) * 0.1)
        
        # 量子设备和电路
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")
        
        # MLR（多层回归）后处理层
        self.mlr = nn.Sequential(
            nn.Linear(n_qubits, HYPERPARAMETERS['MLR_HIDDEN_DIM']),
            nn.ReLU(),
            nn.Linear(HYPERPARAMETERS['MLR_HIDDEN_DIM'], output_dim)
        )
        
    def quantum_circuit(self, params):
        """量子电路定义"""
        # 强纠缠层
        for layer in range(self.n_layers):
            # 数据重上传
            for i in range(self.n_chunks):
                chunk_params = params[i]
                # U3门编码
                qml.U3(chunk_params[0], chunk_params[1], chunk_params[2], wires=i)
            
            # 强纠缠操作
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                
        # 测量Pauli-Z期望值
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.shape[0]
        quantum_outputs = []
        
        # 处理每个样本
        for i in range(batch_size):
            sample = x[i]
            
            # 将输入分割为12个4D块
            chunks = sample.view(self.n_chunks, self.chunk_size)
            
            # 数据重上传：将每个块映射到3D参数
            processed_chunks = []
            for j in range(self.n_chunks):
                chunk = chunks[j]
                # 线性变换：W * x + b -> 3D
                transformed = torch.matmul(self.reupload_weights[j], chunk) + self.reupload_bias[j]
                processed_chunks.append(transformed)
            
            # 量子电路处理
            quantum_params = torch.stack(processed_chunks)
            q_out = self.qnode(quantum_params)
            quantum_outputs.append(torch.stack(q_out))
        
        # 堆叠批次结果
        quantum_output = torch.stack(quantum_outputs)
        
        # 确保数据类型为float32
        quantum_output = quantum_output.float()
        
        # MLR后处理
        final_output = self.mlr(quantum_output)
        
        return final_output

def load_and_preprocess_data(filepath, train_start=HYPERPARAMETERS['TRAIN_START'], 
                           train_end=HYPERPARAMETERS['TRAIN_END']):
    """加载和预处理数据，按指定范围划分为训练集和测试集"""
    # 自动计算测试集范围
    train_samples = train_end - train_start
    test_samples = int(train_samples * HYPERPARAMETERS['TEST_RATIO'])
    test_start = train_end
    test_end = test_start + test_samples
    
    print(f"数据划分: 训练集[{train_start}:{train_end}] ({train_samples}个样本), "
          f"测试集[{test_start}:{test_end}] ({test_samples}个样本)")
    
    # 使用h5py加载MATLAB v7.3文件
    try:
        with h5py.File(filepath, 'r') as f:
            # 尝试常见的数据键名
            data_keys = ['beam_data', 'data', 'X', 'features', 'rsrp']
            data = None
            
            for key in data_keys:
                if key in f:
                    data = np.array(f[key]).T  # MATLAB存储通常是转置的
                    break
            
            # 如果没找到常见键名，尝试第一个数据集
            if data is None:
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data = np.array(f[key]).T
                        break
            
            if data is None:
                raise ValueError("HDF5文件中未找到有效的数据集")
                
    except Exception as e:
        print(f"数据加载错误: {e}")
        raise
    
    # 确保数据是2D的
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    
    # 检查索引范围是否有效（现在在样本维度）
    total_samples = data.shape[1]  # 第二个维度的样本数
    if train_end > total_samples or test_end > total_samples:
        # 调整到有效范围
        train_end = min(train_end, total_samples)
        test_end = min(test_end, total_samples)
        print(f"调整后范围: 训练集[{train_start}:{train_end}], 测试集[{test_start}:{test_end}]")
    
    # 按指定范围选择样本（在第二个维度）
    train_indices = np.arange(train_start, train_end)
    test_indices = np.arange(test_start, test_end)
    
    # 选择样本
    X_train_full = data[:, train_indices].T  # 转置为(样本数, 特征数)
    X_test_full = data[:, test_indices].T
    
    # 等距选择48个特征（在第一个维度）
    n_features = data.shape[0]  # 特征数
    input_indices = np.linspace(0, n_features-1, HYPERPARAMETERS['INPUT_DIM'], dtype=int)
    
    X_train_input = X_train_full[:, input_indices]
    X_test_input = X_test_full[:, input_indices]
    
    # 输出是所有剩余特征
    all_indices = set(range(n_features))
    output_indices = list(all_indices - set(input_indices))
    X_train_output = X_train_full[:, output_indices]
    X_test_output = X_test_full[:, output_indices]
    
    # 数据归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_input_norm = scaler_X.fit_transform(X_train_input)
    X_test_input_norm = scaler_X.transform(X_test_input)
    
    X_train_output_norm = scaler_y.fit_transform(X_train_output)
    X_test_output_norm = scaler_y.transform(X_test_output)
    
    # 转换为PyTorch张量
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

def train_model(model, train_loader, test_loader, device,
                epochs=HYPERPARAMETERS['EPOCHS'], lr=HYPERPARAMETERS['LEARNING_RATE'],
                start_epoch=0):
    """训练模型，包括Early Stopping机制和调试模式支持"""
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
    
    # Early Stopping相关变量
    best_test_loss = float('inf')
    early_stopping_counter = 0
    early_stop_triggered = False
    
    print("开始训练模型...")
    for epoch in range(start_epoch, start_epoch + epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_batch_count = 0
        
        # 调试模式：限制每轮的批次数量
        debug_batch_limit = 3  # 调试模式下限制前3个批次
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 检查调试模式限制
            if HYPERPARAMETERS['DEBUG_MODE'] and batch_idx >= debug_batch_limit:
                break
                
            data, target = data.to(device), target.to(device)  # 移动到指定设备
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batch_count += 1
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / train_batch_count if train_batch_count > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        total_test_loss = 0.0
        total_mae = 0.0
        test_batch_count = 0
        
        with torch.no_grad():
            # 调试模式：限制测试时的批次数量
            debug_batch_limit = 3  # 调试模式下限制前3个批次
            
            for batch_idx, (data, target) in enumerate(test_loader):
                # 检查调试模式限制
                if HYPERPARAMETERS['DEBUG_MODE'] and batch_idx >= debug_batch_limit:
                    break
                    
                data, target = data.to(device), target.to(device)  # 移动到指定设备
                output = model(data)
                
                # 计算MSE损失
                mse_loss = criterion(output, target)
                total_test_loss += mse_loss.item()
                
                # 使用PyTorch计算MAE以保持一致性
                mae = torch.mean(torch.abs(output - target))
                total_mae += mae.item()
                
                test_batch_count += 1
        
        # 计算平均测试指标
        avg_test_loss = total_test_loss / test_batch_count if test_batch_count > 0 else 0.0
        avg_mae = total_mae / test_batch_count if test_batch_count > 0 else 0.0
        
        test_losses.append(avg_test_loss)
        mae_scores.append(avg_mae)
        
        # Early Stopping检查 - 根据监控指标决定是否停止
        if avg_test_loss < best_test_loss - HYPERPARAMETERS['EARLY_STOPPING_MIN_DELTA']:
            best_test_loss = avg_test_loss
            early_stopping_counter = 0
            # 保存最佳模型
            best_model_path = os.path.join(HYPERPARAMETERS['OUTPUT_DIR'], f'best_model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stopping_counter += 1
            
        # 检查是否触发early stopping
        if early_stopping_counter >= HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']:
            print(f"Epoch {epoch}: 触发Early stopping! {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']} 已经 {HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']} 个epochs没有改善。")
            early_stop_triggered = True
            break
        
        # 每10个epoch打印进度
        if epoch % 10 == 0:
            status = " (Early Stop)" if early_stop_triggered else ""
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, MAE: {avg_mae:.6f}{status}')
    
    # 合并历史数据和新训练数据
    if historical_train:
        train_losses, test_losses, mae_scores = merge_training_data(
            historical_train, historical_test, historical_mae,
            train_losses, test_losses, mae_scores
        )
    
    # 训练完成后报告Early Stopping状态
    final_epoch_count = len(train_losses)
    if early_stop_triggered:
        print(f"训练提前停止，总共训练轮数: {final_epoch_count}")
        print(f"最佳 {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']}: {best_test_loss:.6f}")
    else:
        print(f"训练完成 {final_epoch_count} 轮，未触发Early Stopping")
    
    return train_losses, test_losses, mae_scores

def evaluate_model(model, test_loader, scaler_y, device):
    """在测试集上评估模型性能"""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 移动到指定设备
            output = model(data)
            predictions.extend(output.cpu().numpy())  # 移回CPU进行处理
            targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 反变换预测值和目标值
    predictions_original = scaler_y.inverse_transform(predictions)
    targets_original = scaler_y.inverse_transform(targets)
    
    # 计算评估指标
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

def load_latest_model(model, output_dir, device):
    """加载最新的模型参数"""
    model_path, epoch = find_latest_model(output_dir)
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"成功加载模型参数: {model_path} (Epoch {epoch})")
            return epoch
        except Exception as e:
            print(f"加载模型失败: {e}")
            return 0
    else:
        print("未找到可加载的模型文件，从头开始训练")
        return 0

def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='PQC Re-uploading with NPU Support')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'npu'],
                       help='计算设备选择')
    parser.add_argument('--npu', action='store_true', 
                       help='强制使用NPU（等同于--device npu）')
    args = parser.parse_args()
    
    # 设备选择逻辑
    if args.npu:
        device_str = 'npu'
    else:
        device_str = args.device
    
    # 获取最优设备
    device = get_device(device_str)
    print(f"使用设备: {device}")
    
    # 创建输出目录
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = f"{script_name}_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存设备信息
    device_info = {
        'requested_device': device_str,
        'actual_device': str(device),
        'npu_available': NPU_AVAILABLE,
        'cuda_available': torch.cuda.is_available()
    }
    
    device_info_path = os.path.join(output_dir, 'device_info.json')
    with open(device_info_path, 'w') as f:
        json.dump(device_info, f, indent=2)
    print(f"设备信息已保存到: {device_info_path}")
    
    # 数据文件路径
    data_path = HYPERPARAMETERS['DATA_PATH']
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return
    
    # 加载和预处理数据 - 按指定范围划分
    (X_train, y_train, X_test, y_test, scaler_X, scaler_y, 
     input_indices, output_indices, train_indices, test_indices) = load_and_preprocess_data(
        data_path, 
        train_start=HYPERPARAMETERS['TRAIN_START'], 
        train_end=HYPERPARAMETERS['TRAIN_END']
    )
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=HYPERPARAMETERS['BATCH_SIZE'], 
                                             shuffle=HYPERPARAMETERS['SHUFFLE_TRAIN'])
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=HYPERPARAMETERS['BATCH_SIZE'], 
                                            shuffle=False)
    
    # 创建模型
    model = QuantumDataReuploadModel(
        n_qubits=HYPERPARAMETERS['N_QUBITS'], 
        n_layers=HYPERPARAMETERS['N_LAYERS'], 
        input_dim=HYPERPARAMETERS['INPUT_DIM'], 
        output_dim=HYPERPARAMETERS['OUTPUT_DIM']
    ).to(device)  # 将模型移动到指定设备
    
    # 处理是否继续训练的逻辑
    start_epoch = 0
    if HYPERPARAMETERS['CONTINUE_TRAINING']:
        # 尝试加载最新的模型参数以继续训练
        start_epoch = load_latest_model(model, output_dir, device)
    
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
    train_losses, test_losses, mae_scores = train_model(
        model, train_loader, test_loader, device, start_epoch=start_epoch
    )
    
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
        model, test_loader, scaler_y, device
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
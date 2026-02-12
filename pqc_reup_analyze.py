#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PQC_REUP_V1 分析模块
用于生成训练结果的可视化图像和分析报告
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_training_data(epoch_num, output_dir='pqc_reup_v1_output'):
    """加载指定epoch的训练数据，支持累积训练数据"""
    data_path = os.path.join(output_dir, f'training_data_epoch_{epoch_num}.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"训练数据文件不存在: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 检查是否包含训练信息
    if 'training_info' in data:
        training_info = data['training_info']
        print(f"训练信息详情:")
        print(f"  - 起始epoch: {training_info.get('start_epoch', 'N/A')}")
        print(f"  - 最终epoch: {training_info.get('final_epoch', 'N/A')}")
        print(f"  - 历史epochs: {training_info.get('historical_epochs', 'N/A')}")
        print(f"  - 新训练epochs: {training_info.get('new_epochs', 'N/A')}")
        if training_info.get('is_current_run_only', False):
            print(f"  - 数据类型: 仅本次运行数据")
        else:
            print(f"  - 数据类型: 完整累积数据")
    
    return data


def plot_training_curves(train_losses, test_losses, mae_scores, epoch_num, output_dir='pqc_reup_v1_output'):
    """绘制训练曲线，正确显示累积epoch数据"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 计算实际的epoch范围（累积训练）
    total_epochs = len(train_losses)
    epochs = range(1, total_epochs + 1)  # 从1开始，显示实际epoch数
    
    # 损失曲线
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training and Test Loss Curves (Total {total_epochs} Epochs)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(1, total_epochs)  # 设置x轴范围
    
    # MAE曲线
    axes[1].plot(epochs, mae_scores, 'g-', label='MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title(f'Mean Absolute Error Over Time (Total {total_epochs} Epochs)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1, total_epochs)  # 设置x轴范围
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'training_curves_cumulative_{total_epochs}_epochs.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {save_path}")
    return filename

def plot_comprehensive_results(train_losses, test_losses, mae_scores, predictions, targets, input_indices, epoch_num, output_dir='pqc_reup_v1_output'):
    """绘制综合结果图像，包含训练曲线和Top-N分析，正确显示累积epoch数据"""
    # 计算当前epoch的Top-N准确率
    top_n_results = calculate_top_n_accuracy_both_methods(
        predictions, targets, input_indices, top_n_max=10
    )
    
    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 第一行：训练相关曲线
    total_epochs = len(train_losses)
    epochs = range(1, total_epochs + 1)  # 从1开始，显示实际epoch数
    
    # 训练和测试损失曲线
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training and Test Loss Curves (Total {total_epochs} Epochs)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(1, total_epochs)
    
    # MAE曲线
    axes[0, 1].plot(epochs, mae_scores, 'g-', label='MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title(f'Mean Absolute Error Over Time (Total {total_epochs} Epochs)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(1, total_epochs)
    
    # Top-N准确率对比曲线（当前epoch的准确率）
    n_values = list(range(1, len(top_n_results['with_input']) + 1))
    axes[0, 2].plot(n_values, top_n_results['with_input'], 'o-', linewidth=2, markersize=6, 
                    label='Including Input Beams', color='blue')
    axes[0, 2].plot(n_values, top_n_results['without_input'], 's-', linewidth=2, markersize=6, 
                    label='Excluding Input Beams', color='red')
    axes[0, 2].set_xlabel('N')
    axes[0, 2].set_ylabel('Top-N Accuracy')
    axes[0, 2].set_title(f'Top-N Accuracy Comparison')
    axes[0, 2].set_xticks(n_values)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 第二行：预测分析和Top-N随epoch变化曲线
    sample_size = min(1000, len(predictions))
    sample_indices = np.random.choice(len(predictions), sample_size, replace=False)
    
    # 预测vs真实值散点图
    axes[1, 0].scatter(targets[sample_indices].flatten(), predictions[sample_indices].flatten(), 
                      alpha=0.5, s=1)
    axes[1, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predictions')
    axes[1, 0].set_title(f'Predictions vs True Values (Sample)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 误差分布
    errors = (predictions - targets).flatten()
    axes[1, 1].hist(errors, bins=50, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Top-N准确率随epoch变化曲线（只显示包含输入波束的方法）
    # 收集所有可用epoch的Top-N数据
    top_n_evolution_with = []    # 方法A各epoch的Top-N准确率
    evolution_epochs = []        # 对应的epoch编号
    
    # 遍历所有可用的epoch数据文件
    for epoch_check in range(1, epoch_num + 1):
        eval_file = os.path.join(output_dir, f'evaluation_results_epoch_{epoch_check}.json')
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                pred_check = np.array(eval_data['predictions'])
                target_check = np.array(eval_data['targets'])
                
                # 计算该epoch的Top-N准确率（只计算包含输入波束的方法）
                epoch_results = calculate_top_n_accuracy_both_methods(
                    pred_check, target_check, input_indices, top_n_max=10
                )
                
                top_n_evolution_with.append(epoch_results['with_input'])
                evolution_epochs.append(epoch_check)
                
            except Exception as e:
                print(f"警告: 无法加载epoch {epoch_check} 的评估数据: {e}")
                continue
    
    # 绘制Top-N准确率演化曲线（只显示包含输入波束）
    if top_n_evolution_with:
        n_count = len(top_n_evolution_with[0])  # N的数量
        # 创建从红色到蓝色的颜色过渡
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, n_count))  # 反转色彩映射使Top-1为红色
        
        # 为每个Top-N值绘制演化曲线
        legend_lines = []
        legend_labels = []
        for n_idx in range(n_count):
            # 方法A：包含输入波束 - 实线
            with_acc_by_epoch = [epoch_data[n_idx] for epoch_data in top_n_evolution_with]
            line = axes[1, 2].plot(evolution_epochs, with_acc_by_epoch, 
                                  linewidth=2, linestyle='-', alpha=0.8,
                                  color=colors[n_idx])
            
            legend_lines.append(line[0])
            legend_labels.append(f'Top-{n_idx+1}')
        
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Top-N Accuracy')
        axes[1, 2].set_title(f'Top-N Accuracy Evolution (Including Input Beams)')
        axes[1, 2].set_ylim(0, 1)
        
        # 将图例放在图像内部左上角
        axes[1, 2].legend(legend_lines, legend_labels, 
                         loc='upper left', bbox_to_anchor=(0.02, 0.98),
                         frameon=True, fancybox=True, shadow=True)
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # 如果没有历史数据，显示当前epoch的水平线
        n_range = range(1, len(top_n_results['with_input']) + 1)
        # 创建从红色到蓝色的颜色过渡
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(n_range)))
        
        # 为所有Top-N值显示水平线
        legend_lines = []
        legend_labels = []
        for i, acc_with in enumerate(top_n_results['with_input']):
            line = axes[1, 2].axhline(y=acc_with, xmin=0, xmax=1, color=colors[i], 
                                     linewidth=2, linestyle='-', alpha=0.7)
            legend_lines.append(line)
            legend_labels.append(f'Top-{i+1}')
        
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Top-N Accuracy')
        axes[1, 2].set_title(f'Top-N Accuracy (Current Epoch Only)')
        axes[1, 2].set_ylim(0, 1)
        
        # 将图例放在图像内部左上角
        axes[1, 2].legend(legend_lines, legend_labels, 
                         loc='upper left', bbox_to_anchor=(0.02, 0.98),
                         frameon=True, fancybox=True, shadow=True)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'results_cumulative_{total_epochs}_epochs.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"综合结果图像已保存到: {save_path}")
    return filename

def plot_prediction_analysis(predictions, targets, epoch_num, output_dir='pqc_reup_v1_output'):
    """绘制预测分析图像"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 预测vs真实值散点图（采样显示）
    sample_size = min(1000, len(predictions))
    sample_indices = np.random.choice(len(predictions), sample_size, replace=False)
    
    axes[0, 0].scatter(targets[sample_indices].flatten(), predictions[sample_indices].flatten(), 
                      alpha=0.5, s=1)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predictions')
    axes[0, 0].set_title(f'Predictions vs True Values (Sample) - Epoch {epoch_num}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 误差分布
    errors = (predictions - targets).flatten()
    axes[0, 1].hist(errors, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Error Distribution - Epoch {epoch_num}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差图
    axes[1, 0].scatter(predictions[sample_indices].flatten(), errors[sample_indices], 
                      alpha=0.5, s=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title(f'Residual Plot - Epoch {epoch_num}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q图（简化版）
    sorted_errors = np.sort(errors)
    theoretical_quantiles = np.linspace(sorted_errors.min(), sorted_errors.max(), len(sorted_errors))
    axes[1, 1].scatter(theoretical_quantiles, sorted_errors, alpha=0.5, s=1)
    axes[1, 1].plot([theoretical_quantiles.min(), theoretical_quantiles.max()], 
                   [theoretical_quantiles.min(), theoretical_quantiles.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Theoretical Quantiles')
    axes[1, 1].set_ylabel('Sample Quantiles')
    axes[1, 1].set_title(f'Q-Q Plot (Approximate) - Epoch {epoch_num}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'prediction_analysis_epoch_{epoch_num}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"预测分析图像已保存到: {save_path}")
    return filename

def calculate_top_n_accuracy_both_methods(predictions, targets, input_indices, top_n_max=10):
    """计算两种Top-N准确率：包含输入波束和不包含输入波束"""
    n_samples = len(predictions)
    
    # 初始化两种方法的准确率计数器
    top_n_correct_with_input = [0] * top_n_max    # 方法A：包含输入波束
    top_n_correct_without_input = [0] * top_n_max # 方法B：不包含输入波束
    
    # 创建输出波束索引集合（排除输入波束）
    all_indices = set(range(len(predictions[0])))  # 所有波束索引
    output_indices_set = all_indices - set(input_indices)  # 排除输入波束后的索引
    
    for i in range(n_samples):
        pred_sample = predictions[i]
        target_sample = targets[i]
        
        # 方法A：包含输入波束的统计
        pred_indices_A = np.argsort(pred_sample)[::-1]  # 所有波束降序排列
        target_max_idx_A = np.argmax(target_sample)     # 真实最大值索引
        
        # 方法B：不包含输入波束的统计
        pred_values_B = pred_sample[list(output_indices_set)]
        target_values_B = target_sample[list(output_indices_set)]
        
        # 获取输出波束内的排序索引
        pred_local_indices_B = np.argsort(pred_values_B)[::-1]  # 输出波束降序排列
        target_local_max_idx_B = np.argmax(target_values_B)     # 真实最大值在输出波束内的索引
        
        # 将局部索引映射回全局索引
        output_indices_list = list(output_indices_set)
        pred_global_indices_B = [output_indices_list[idx] for idx in pred_local_indices_B]
        target_global_max_idx_B = output_indices_list[target_local_max_idx_B]
        
        # 计算两种方法的Top-N准确率
        for n in range(1, top_n_max + 1):
            # 方法A：检查真实最优波束是否在预测的前N个中（包含所有波束）
            if target_max_idx_A in pred_indices_A[:n]:
                top_n_correct_with_input[n-1] += 1
            
            # 方法B：检查真实最优波束是否在预测的前N个中（仅输出波束）
            if target_global_max_idx_B in pred_global_indices_B[:n]:
                top_n_correct_without_input[n-1] += 1
    
    # 计算准确率
    top_n_accuracies_with_input = [correct / n_samples for correct in top_n_correct_with_input]
    top_n_accuracies_without_input = [correct / n_samples for correct in top_n_correct_without_input]
    
    return {
        'with_input': top_n_accuracies_with_input,      # 方法A
        'without_input': top_n_accuracies_without_input  # 方法B
    }

def plot_top_n_analysis(predictions, targets, input_indices, epoch_num, output_dir='pqc_reup_v1_output'):
    """绘制Top-N准确率分析图像"""
    # 计算当前epoch的Top-N准确率
    top_n_results = calculate_top_n_accuracy_both_methods(
        predictions, targets, input_indices, top_n_max=10
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top-N准确率对比曲线（按N值）
    n_values = list(range(1, len(top_n_results['with_input']) + 1))
    
    # 方法A：包含输入波束
    axes[0].plot(n_values, top_n_results['with_input'], 'o-', linewidth=2, markersize=6, 
                 label='Including Input Beams', color='blue')
    
    # 方法B：不包含输入波束（仍然显示用于对比）
    axes[0].plot(n_values, top_n_results['without_input'], 's-', linewidth=2, markersize=6, 
                 label='Excluding Input Beams', color='red')
    
    axes[0].set_xlabel('N')
    axes[0].set_ylabel('Top-N Accuracy')
    axes[0].set_title(f'Top-N Accuracy by N Value - Epoch {epoch_num}')
    axes[0].set_xticks(n_values)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Top-N准确率随epoch变化曲线（只显示包含输入波束的方法）
    # 收集所有可用epoch的Top-N数据
    top_n_evolution_with = []    # 方法A各epoch的Top-N准确率
    evolution_epochs = []        # 对应的epoch编号
    
    # 遍历所有可用的epoch数据文件
    for epoch_check in range(1, epoch_num + 1):
        eval_file = os.path.join(output_dir, f'evaluation_results_epoch_{epoch_check}.json')
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                pred_check = np.array(eval_data['predictions'])
                target_check = np.array(eval_data['targets'])
                
                # 计算该epoch的Top-N准确率（只计算包含输入波束的方法）
                epoch_results = calculate_top_n_accuracy_both_methods(
                    pred_check, target_check, input_indices, top_n_max=10
                )
                
                top_n_evolution_with.append(epoch_results['with_input'])
                evolution_epochs.append(epoch_check)
                
            except Exception as e:
                print(f"警告: 无法加载epoch {epoch_check} 的评估数据: {e}")
                continue
    
    # 绘制Top-N准确率演化曲线（只显示包含输入波束）
    if top_n_evolution_with:
        n_count = len(top_n_evolution_with[0])  # N的数量
        # 创建从红色到蓝色的颜色过渡
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, n_count))  # 反转色彩映射使Top-1为红色
        
        # 为每个Top-N值绘制演化曲线
        legend_lines = []
        legend_labels = []
        for n_idx in range(n_count):
            # 方法A：包含输入波束 - 实线
            with_acc_by_epoch = [epoch_data[n_idx] for epoch_data in top_n_evolution_with]
            line = axes[1].plot(evolution_epochs, with_acc_by_epoch, 
                               linewidth=2, linestyle='-', alpha=0.8,
                               color=colors[n_idx])
            
            legend_lines.append(line[0])
            legend_labels.append(f'Top-{n_idx+1}')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Top-N Accuracy')
        axes[1].set_title(f'Top-N Accuracy Evolution (Including Input Beams)')
        axes[1].set_ylim(0, 1)
        
        # 将图例放在图像内部左上角
        axes[1].legend(legend_lines, legend_labels, 
                      loc='upper left', bbox_to_anchor=(0.02, 0.98),
                      frameon=True, fancybox=True, shadow=True)
        axes[1].grid(True, alpha=0.3)
    else:
        # 如果没有历史数据，显示当前epoch的水平线
        n_range = range(1, len(top_n_results['with_input']) + 1)
        # 创建从红色到蓝色的颜色过渡
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(n_range)))
        
        # 为所有Top-N值显示水平线
        legend_lines = []
        legend_labels = []
        for i, acc_with in enumerate(top_n_results['with_input']):
            line = axes[1].axhline(y=acc_with, xmin=0, xmax=1, color=colors[i], 
                                  linewidth=2, linestyle='-', alpha=0.7)
            legend_lines.append(line)
            legend_labels.append(f'Top-{i+1}')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Top-N Accuracy')
        axes[1].set_title(f'Top-N Accuracy (Current Epoch Only)')
        axes[1].set_ylim(0, 1)
        
        # 将图例放在图像内部左上角
        axes[1].legend(legend_lines, legend_labels, 
                      loc='upper left', bbox_to_anchor=(0.02, 0.98),
                      frameon=True, fancybox=True, shadow=True)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'top_n_analysis_epoch_{epoch_num}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Top-N分析图像已保存到: {save_path}")
    return filename

def generate_analysis_report(training_data, predictions, targets, epoch_num, input_indices=None, output_dir='pqc_reup_v1_output'):
    """生成分析报告"""
    # 计算评估指标
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    
    # 计算R²分数
    ss_res = np.sum((targets.flatten() - predictions.flatten()) ** 2)
    ss_tot = np.sum((targets.flatten() - np.mean(targets.flatten())) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 计算皮尔逊相关系数
    correlation, _ = pearsonr(targets.flatten(), predictions.flatten())
    
    # 性能等级评定
    if r2 > 0.75:
        performance_level = "Excellent"
    elif r2 > 0.6:
        performance_level = "Good"
    elif r2 > 0.3:
        performance_level = "Fair"
    elif r2 > 0.1:
        performance_level = "Poor"
    else:
        performance_level = "Very Poor"
    
    # 生成报告内容
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""# PQC_REUP_V1 Analysis Report - Epoch {epoch_num}

## Experiment Information
- **Analysis Time**: {timestamp}
- **Trained Epochs**: {epoch_num}
- **Output Directory**: {output_dir}

## Performance Metrics
- **MSE (Mean Squared Error)**: {mse:.6f}
- **MAE (Mean Absolute Error)**: {mae:.6f}
- **RMSE (Root Mean Squared Error)**: {rmse:.6f}
- **R² Score**: {r2:.6f}
- **Pearson Correlation**: {correlation:.6f}
- **Performance Level**: {performance_level}

## Training Statistics
- **Final Training Loss**: {training_data['train_losses'][-1]:.6f}
- **Final Test Loss**: {training_data['test_losses'][-1]:.6f}
- **Final MAE**: {training_data['mae_scores'][-1]:.6f}
- **Best Test Loss**: {min(training_data['test_losses']):.6f}
- **Best MAE**: {min(training_data['mae_scores']):.6f}

## Generated Files
- Training curves plot: training_curves_epoch_{epoch_num}.png
- Prediction analysis plot: prediction_analysis_epoch_{epoch_num}.png
- This analysis report: analysis_report_epoch_{epoch_num}.md

## Notes
This analysis was automatically generated by pqc_reup_analyze.py
"""
    
    # 保存报告
    report_filename = f'analysis_report_epoch_{epoch_num}.md'
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"分析报告已保存到: {report_path}")
    return report_filename

def analyze_results(epoch_num, predictions=None, targets=None, input_indices=None, output_dir='pqc_reup_v1_output', **kwargs):
    """主分析函数 - 读取最新的合并训练过程数据文件并绘图"""
    print(f"开始分析第 {epoch_num} 轮训练结果...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载训练数据（直接加载指定epoch的合并数据文件）
    try:
        training_data = load_training_data(epoch_num, output_dir)
        total_epochs = len(training_data['train_losses'])
        print(f"成功加载训练数据 ({total_epochs} 个数据点)")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 如果提供了预测、目标和输入索引数据，则进行完整分析
    if predictions is not None and targets is not None and input_indices is not None:
        print("生成完整分析报告...")
        
        # 使用新的综合图像生成函数（包含训练曲线、预测分析和Top-N准确率）
        results_filename = plot_comprehensive_results(
            training_data['train_losses'],
            training_data['test_losses'],
            training_data['mae_scores'],
            predictions,
            targets,
            input_indices,
            total_epochs,  # 使用累积epoch数量
            output_dir
        )
        
        # 生成分析报告
        report_filename = generate_analysis_report(
            training_data, predictions, targets, total_epochs, input_indices, output_dir
        )
        
        print(f"\n分析完成！生成的文件:")
        print(f"- {results_filename}")
        print(f"- {report_filename}")
    else:
        print("仅生成训练曲线分析...")
        # 只绘制训练曲线
        curve_filename = plot_training_curves(
            training_data['train_losses'],
            training_data['test_losses'],
            training_data['mae_scores'],
            total_epochs,  # 使用累积epoch数量
            output_dir
        )
        print(f"训练曲线已保存: {curve_filename}")

def main():
    """主函数 - 支持命令行运行并自动生成results_epoch_x.png"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PQC_REUP_V1 结果分析工具')
    parser.add_argument('--epoch', type=int, default=None, 
                       help='要分析的epoch编号（默认自动检测最新）')
    parser.add_argument('--output-dir', type=str, default='pqc_reup_v1_output',
                       help='输出目录路径')
    parser.add_argument('--input-indices', type=str, default=None,
                       help='输入波束索引，逗号分隔（如："0,1,2,3"）')
    parser.add_argument('--top-n-max', type=int, default=10,
                       help='Top-N分析的最大N值（默认10）')
    args = parser.parse_args()
    
    print("PQC_REUP_V1 结果分析工具")
    print("=" * 50)
    
    # 确定要分析的epoch
    if args.epoch is not None:
        epoch_to_analyze = args.epoch
        print(f"分析指定epoch: {epoch_to_analyze}")
    else:
        # 自动检测最新的epoch
        epoch_to_analyze = find_latest_epoch(args.output_dir)
        if epoch_to_analyze is None:
            print("错误: 未找到任何训练数据文件")
            return
        print(f"自动检测到最新epoch: {epoch_to_analyze}")
    
    # 处理输入索引参数
    input_indices = None
    if args.input_indices:
        try:
            input_indices = [int(x.strip()) for x in args.input_indices.split(',')]
            print(f"使用指定输入索引: {input_indices}")
        except Exception as e:
            print(f"警告: 输入索引格式错误: {e}")
            print("使用默认输入索引 [0, 1, 2, ..., 11]")
            input_indices = list(range(12))
    else:
        # 尝试从配置文件获取输入索引
        config_files = [f for f in os.listdir(args.output_dir) if f.startswith('config_') and f.endswith('.md')]
        if config_files:
            config_files.sort(reverse=True)
            latest_config = config_files[0]
            config_path = os.path.join(args.output_dir, latest_config)
            try:
                with open(config_path, 'r') as f:
                    config_content = f.read()
                    import re
                    indices_match = re.search(r'input_indices.*?\[(.*?)\]', config_content)
                    if indices_match:
                        indices_str = indices_match.group(1)
                        input_indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
                        print(f"从配置文件获取输入索引: {input_indices}")
            except Exception as e:
                print(f"警告: 无法从配置文件解析输入索引: {e}")
        
        if input_indices is None:
            print("使用默认输入索引 [0, 1, 2, ..., 11]")
            input_indices = list(range(12))
    
    # 检查评估结果文件
    evaluation_file = os.path.join(args.output_dir, f'evaluation_results_epoch_{epoch_to_analyze}.json')
    
    if os.path.exists(evaluation_file):
        print(f"检测到评估结果文件: {evaluation_file}")
        try:
            # 加载评估结果
            with open(evaluation_file, 'r') as f:
                eval_data = json.load(f)
            
            predictions = np.array(eval_data['predictions'])
            targets = np.array(eval_data['targets'])
            
            print(f"加载预测数据: {predictions.shape[0]} 个样本, {predictions.shape[1]} 个输出维度")
            print(f"加载真实数据: {targets.shape[0]} 个样本, {targets.shape[1]} 个输出维度")
            
            # 进行完整分析（包含Top-N accuracy统计）
            print(f"\n开始完整分析 (Top-N最大值: {args.top_n_max})...")
            
            # 计算Top-N准确率
            top_n_results = calculate_top_n_accuracy_both_methods(
                predictions, targets, input_indices, top_n_max=args.top_n_max
            )
            
            # 显示Top-N准确率结果
            print("\nTop-N 准确率统计:")
            print("-" * 50)
            print(f"{'方法':<15} {'N值':<8} {'准确率':<12} {'百分比':<10}")
            print("-" * 50)
            
            # 方法A：包含输入波束
            for i, acc in enumerate(top_n_results['with_input']):
                print(f"{'包含输入波束':<15} {f'Top-{i+1}':<8} {acc:<12.6f} {acc*100:<10.2f}%")
            
            print("-" * 50)
            
            # 方法B：不包含输入波束
            for i, acc in enumerate(top_n_results['without_input']):
                print(f"{'排除输入波束':<15} {f'Top-{i+1}':<8} {acc:<12.6f} {acc*100:<10.2f}%")
            
            # 计算平均准确率
            avg_with_input = np.mean(top_n_results['with_input'])
            avg_without_input = np.mean(top_n_results['without_input'])
            print("-" * 50)
            print(f"{'平均准确率':<15} {'':<8} {avg_with_input:<12.6f} {avg_with_input*100:<10.2f}% (包含输入)")
            print(f"{'平均准确率':<15} {'':<8} {avg_without_input:<12.6f} {avg_without_input*100:<10.2f}% (排除输入)")
            
            # 加载训练数据
            try:
                training_data = load_training_data(epoch_to_analyze, args.output_dir)
                print(f"\n成功加载训练数据 ({len(training_data['train_losses'])} 个数据点)")
                
                # 生成综合分析报告
                print("\n生成综合分析图像...")
                results_filename = plot_comprehensive_results(
                    training_data['train_losses'],
                    training_data['test_losses'],
                    training_data['mae_scores'],
                    predictions,
                    targets,
                    input_indices,
                    epoch_to_analyze,
                    args.output_dir
                )
                
                # 生成分析报告
                report_filename = generate_analysis_report(
                    training_data, predictions, targets, epoch_to_analyze, input_indices, args.output_dir
                )
                
                print(f"\n分析完成！生成的文件:")
                print(f"- 综合结果图像: {results_filename}")
                print(f"- 分析报告: {report_filename}")
                
            except Exception as e:
                print(f"警告: 训练数据加载失败: {e}")
                print("仅生成Top-N分析结果")
                
        except Exception as e:
            print(f"错误: 加载评估数据失败: {e}")
            print("请检查评估结果文件格式")
    else:
        print(f"未找到评估结果文件: {evaluation_file}")
        print("仅进行训练曲线分析...")
        
        # 尝试进行基础训练曲线分析
        try:
            training_data = load_training_data(epoch_to_analyze, args.output_dir)
            curve_filename = plot_training_curves(
                training_data['train_losses'],
                training_data['test_losses'],
                training_data['mae_scores'],
                epoch_to_analyze,
                args.output_dir
            )
            print(f"训练曲线已保存: {curve_filename}")
        except Exception as e:
            print(f"错误: 基础分析失败: {e}")

def find_latest_epoch(output_dir):
    """查找最新的epoch编号"""
    if not os.path.exists(output_dir):
        return None
    
    # 查找训练数据文件
    data_files = [f for f in os.listdir(output_dir) if f.startswith('training_data_epoch_') and f.endswith('.json')]
    
    if not data_files:
        return None
    
    # 提取epoch编号
    epochs = []
    for f in data_files:
        try:
            epoch = int(f.split('_')[3].split('.')[0])
            epochs.append(epoch)
        except:
            continue
    
    return max(epochs) if epochs else None

if __name__ == "__main__":
    main()
# distill.py - 用于双流YOLOv8网络的蒸馏训练脚本

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER, colorstr


class TwoStreamDistillationLoss(nn.Module):
    """双流网络的蒸馏损失"""
    def __init__(self, T=3.0, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_outputs, teacher_outputs):
        """
        计算教师和学生模型输出之间的KL散度
        student_outputs: 学生模型输出的预测
        teacher_outputs: 教师模型输出的预测
        """
        # 针对YOLOv8的输出格式进行处理
        # YOLOv8输出格式: [batch_size, num_anchors, grid_h, grid_w, 5+num_classes]
        
        # 提取边界框预测和置信度分数
        student_boxes = student_outputs[..., :4]
        student_conf = student_outputs[..., 4:5]
        student_cls = student_outputs[..., 5:]
        
        teacher_boxes = teacher_outputs[..., :4]
        teacher_conf = teacher_outputs[..., 4:5]
        teacher_cls = teacher_outputs[..., 5:]
        
        # 对边界框位置进行蒸馏
        box_loss = F.mse_loss(student_boxes, teacher_boxes)
        
        # 对置信度分数进行蒸馏
        conf_loss = F.mse_loss(student_conf, teacher_conf)
        
        # 对类别预测进行蒸馏（使用温度调整的KL散度）
        soft_student = F.log_softmax(student_cls / self.T, dim=-1)
        soft_teacher = F.softmax(teacher_cls / self.T, dim=-1)
        cls_loss = self.kl_div(soft_student, soft_teacher) * (self.T ** 2)
        
        # 总蒸馏损失
        distill_loss = box_loss + conf_loss + cls_loss
        return distill_loss * self.alpha


class TwoStreamFeatureDistillationLoss(nn.Module):
    """双流网络的特征蒸馏损失"""
    def __init__(self, student_channels, teacher_channels, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        # 通道适配层
        self.adapt = nn.Identity() if student_channels == teacher_channels else nn.Conv2d(student_channels, teacher_channels, 1)
        
    def forward(self, student_feat, teacher_feat):
        """计算特征蒸馏损失"""
        # 调整学生特征的通道数以匹配教师特征
        student_feat = self.adapt(student_feat)
        
        # 如果空间维度不同，则调整学生特征的空间维度
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.interpolate(student_feat, teacher_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 计算特征蒸馏损失（使用MSE）
        return F.mse_loss(student_feat, teacher_feat) * self.alpha


class TwoStreamYOLODistiller:
    """专为双流YOLOv8设计的蒸馏器"""
    def __init__(self, teacher_model, student_model_cfg, 
                 feat_loss_weight=0.5, logit_loss_weight=0.5, 
                 temperature=3.0, device='cuda'):
        self.device = device
        
        # 加载教师模型
        self.teacher = teacher_model.to(device)
        self.teacher.eval()
        
        # 冻结教师模型参数
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 创建学生模型（从YAML配置）
        self.student = self._create_student_model(student_model_cfg).to(device)
        
        # 注册钩子以获取中间特征
        self.teacher_features = {}
        self.student_features = {}
        self._register_hooks()
        
        # 初始化特征蒸馏损失
        self.feature_losses = []
        self._init_feature_losses(feat_loss_weight)
        
        # 初始化输出蒸馏损失
        self.output_loss = TwoStreamDistillationLoss(T=temperature, alpha=logit_loss_weight)
        
        LOGGER.info(f"{colorstr('bright_cyan', 'bold', 'Two-Stream YOLO Distiller:')} initialized "
                    f"with teacher model: {teacher_model.__class__.__name__} "
                    f"and student config: {student_model_cfg}")
    
    def _create_student_model(self, cfg):
        """从YAML配置创建学生模型"""
        # 如果是轻量化版本的YAML配置，则直接使用
        if isinstance(cfg, str) and cfg.endswith('.yaml'):
            return DetectionModel(cfg)
        # 否则使用标准模型加载
        else:
            return YOLO(cfg).model
    
    def _register_hooks(self):
        """注册钩子以捕获中间特征"""
        def get_teacher_hook(name):
            def hook(module, input, output):
                self.teacher_features[name] = output
            return hook
        
        def get_student_hook(name):
            def hook(module, input, output):
                self.student_features[name] = output
            return hook
        
        # 注册教师模型中间特征的钩子
        # 针对双流网络架构，我们监控融合前后的关键特征
        
        # 监控教师模型的骨干网络
        for i, module in enumerate(self.teacher.model.backbone.children()):
            if isinstance(module, nn.Module) and any(x in str(type(module)) for x in ['C2f', 'SPPF', 'Conv']):
                module.register_forward_hook(get_teacher_hook(f"backbone.{i}"))
        
        # 监控教师模型的头部网络
        for i, module in enumerate(self.teacher.model.head.children()):
            if isinstance(module, nn.Module) and any(x in str(type(module)) for x in ['C2f', 'Conv']):
                module.register_forward_hook(get_teacher_hook(f"head.{i}"))
        
        # 监控学生模型的骨干网络
        for i, module in enumerate(self.student.model.backbone.children()):
            if isinstance(module, nn.Module) and any(x in str(type(module)) for x in ['C2f', 'SPPF', 'Conv']):
                module.register_forward_hook(get_student_hook(f"backbone.{i}"))
        
        # 监控学生模型的头部网络
        for i, module in enumerate(self.student.model.head.children()):
            if isinstance(module, nn.Module) and any(x in str(type(module)) for x in ['C2f', 'Conv']):
                module.register_forward_hook(get_student_hook(f"head.{i}"))
    
    def _init_feature_losses(self, alpha):
        """初始化特征蒸馏损失"""
        # 首先进行一次前向传播以获取特征维度
        dummy_input = torch.randn(1, 6, 640, 640).to(self.device)  # 假设双流输入为6通道
        with torch.no_grad():
            self.teacher(dummy_input)
        self.student(dummy_input)
        
        # 为匹配的特征创建蒸馏损失
        matched_features = []
        for t_name, t_feat in self.teacher_features.items():
            # 找到匹配的学生特征
            best_match = None
            for s_name, s_feat in self.student_features.items():
                if t_name.split('.')[-1] == s_name.split('.')[-1]:
                    best_match = (s_name, s_feat)
                    break
            
            if best_match:
                s_name, s_feat = best_match
                # 创建特征蒸馏损失
                self.feature_losses.append((
                    (t_name, s_name),
                    TwoStreamFeatureDistillationLoss(
                        student_channels=s_feat.shape[1],
                        teacher_channels=t_feat.shape[1],
                        alpha=alpha
                    ).to(self.device)
                ))
                matched_features.append((t_name, s_name))
        
        LOGGER.info(f"Initialized {len(self.feature_losses)} feature distillation losses")
        for t_name, s_name in matched_features:
            LOGGER.debug(f"  Teacher: {t_name} -> Student: {s_name}")
    
    def compute_distillation_loss(self, images, targets):
        """计算蒸馏损失"""
        # 清空特征缓存
        self.teacher_features.clear()
        self.student_features.clear()
        
        # 前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher(images)
        
        student_outputs = self.student(images)
        
        # 计算标准任务损失
        task_loss = self.student.criterion(student_outputs, targets)
        
        # 计算特征蒸馏损失
        feat_loss = 0
        for (t_name, s_name), loss_fn in self.feature_losses:
            if t_name in self.teacher_features and s_name in self.student_features:
                feat_loss += loss_fn(self.student_features[s_name], self.teacher_features[t_name])
        
        # 计算输出蒸馏损失
        output_loss = self.output_loss(student_outputs[0], teacher_outputs[0])
        
        # 总损失
        total_loss = task_loss + feat_loss + output_loss
        
        return total_loss, task_loss, feat_loss, output_loss
    
    def train_one_epoch(self, dataloader, optimizer, scheduler=None, epoch=0):
        """训练一个轮次"""
        self.student.train()
        total_loss = 0
        total_task_loss = 0
        total_feat_loss = 0
        total_output_loss = 0
        
        for i, batch in enumerate(dataloader):
            images, targets = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 计算蒸馏损失
            loss, task_loss, feat_loss, output_loss = self.compute_distillation_loss(images, targets)
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 学习率调度
            if scheduler:
                scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_feat_loss += feat_loss.item()
            total_output_loss += output_loss.item()
            
            # 打印进度
            if i % 10 == 0:
                LOGGER.info(f"Epoch: {epoch}, Batch: {i}, "
                           f"Loss: {loss.item():.4f}, Task: {task_loss.item():.4f}, "
                           f"Feat: {feat_loss.item():.4f}, Output: {output_loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        avg_task_loss = total_task_loss / len(dataloader)
        avg_feat_loss = total_feat_loss / len(dataloader)
        avg_output_loss = total_output_loss / len(dataloader)
        
        LOGGER.info(f"Epoch {epoch} Summary - "
                   f"Avg Loss: {avg_loss:.4f}, Avg Task: {avg_task_loss:.4f}, "
                   f"Avg Feat: {avg_feat_loss:.4f}, Avg Output: {avg_output_loss:.4f}")
        
        return avg_loss
    
    def save_student(self, path):
        """保存学生模型"""
        torch.save(self.student.state_dict(), path)
        LOGGER.info(f"Student model saved to {path}")
    
    def export_student(self, format='onnx'):
        """导出学生模型为推理格式"""
        # 实现模型导出逻辑
        pass


# 命令行接口
def parse_args():
    parser = argparse.ArgumentParser(description='Two-Stream YOLOv8 Knowledge Distillation')
    parser.add_argument('--teacher', type=str, required=True, help='Teacher model path or name')
    parser.add_argument('--student', type=str, required=True, help='Student model YAML config')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML config')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--device', default='', help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--feat-weight', type=float, default=0.5, help='Feature distillation loss weight')
    parser.add_argument('--logit-weight', type=float, default=0.5, help='Logit distillation loss weight')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--output-dir', type=str, default='runs/distill', help='Output directory')
    return parser.parse_args()


# 主函数
def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载教师模型
    LOGGER.info(f"Loading teacher model from {args.teacher}")
    teacher_model = YOLO(args.teacher)
    
    # 设置设备
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化蒸馏器
    distiller = TwoStreamYOLODistiller(
        teacher_model=teacher_model,
        student_model_cfg=args.student,
        feat_loss_weight=args.feat_weight,
        logit_loss_weight=args.logit_weight,
        temperature=args.temperature,
        device=device
    )
    
    # 创建数据加载器
    # 这里使用YOLO的数据加载逻辑...
    # dataloader = ...
    
    # 创建优化器
    optimizer = torch.optim.Adam(distiller.student.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    for epoch in range(args.epochs):
        LOGGER.info(f"Starting epoch {epoch}")
        
        # 训练一个轮次
        distiller.train_one_epoch(dataloader, optimizer, scheduler, epoch)
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"student_epoch_{epoch}.pt")
            distiller.save_student(checkpoint_path)
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "student_final.pt")
    distiller.save_student(final_path)
    
    LOGGER.info(f"Distillation training completed! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
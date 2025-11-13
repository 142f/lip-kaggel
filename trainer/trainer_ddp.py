import os
import torch
import torch.nn as nn
import torch.distributed as dist
from models import build_model, get_loss


class TrainerDDP(nn.Module):
    """支持DDP分布式训练的Trainer类"""
    
    def __init__(self, opt, rank):
        super().__init__()
        self.rank = rank
        self.opt = opt
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{rank}")
        
        # 训练步数追踪
        self.total_steps = 0
        self.update_steps = 0
        
        # 保存目录
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 构建模型
        self.model = build_model(opt.arch)
        self.model = self.model.to(self.device)
        
        # 处理预训练权重加载和step_bias
        self._load_pretrained_weights(opt)
        
        # 冻结编码器（如需要）
        self._setup_trainable_params(opt)
        
        # 创建优化器
        self.optimizer = self._create_optimizer(opt)
        
        # 学习率调度器
        self.scheduler = self._create_scheduler(opt) if opt.cosine_annealing else None
        
        # 损失函数
        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss().to(self.device)
        
        # 梯度累积相关参数
        self.accumulation_steps = getattr(opt, 'accumulation_steps', 1)
        self.accumulation_count = 0
    
    def _load_pretrained_weights(self, opt):
        """加载预训练权重"""
        self.step_bias = 0
        
        if opt.fine_tune and hasattr(opt, 'pretrained_model') and opt.pretrained_model:
            try:
                state_dict = torch.load(opt.pretrained_model, map_location="cpu")
                self.model.load_state_dict(state_dict.get("model", state_dict), strict=False)
                self.total_steps = state_dict.get("total_steps", 0)
                self.step_bias = int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
                
                if self.rank == 0:
                    print(f"✓ 预训练模型已加载: {opt.pretrained_model.split('/')[-1]}")
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠ 加载预训练权重失败: {e}")
    
    def _setup_trainable_params(self, opt):
        """设置可训练参数"""
        if hasattr(opt, 'fix_encoder') and opt.fix_encoder:
            for name, param in self.model.named_parameters():
                if name.split(".")[0] == "encoder":
                    param.requires_grad = False
    
    def _create_optimizer(self, opt):
        """创建优化器"""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if opt.optim == "adamw":
            return torch.optim.AdamW(
                trainable_params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "adam":
            return torch.optim.Adam(
                trainable_params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            return torch.optim.SGD(
                trainable_params,
                lr=opt.lr,
                momentum=0.9,
                weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim must be one of: [sgd, adam, adamw]")
    
    def _create_scheduler(self, opt):
        """创建学习率调度器"""
        if not opt.cosine_annealing:
            return None
        
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=getattr(opt, 'scheduler_T0', 20),
            T_mult=1,
            eta_min=getattr(opt, 'scheduler_eta_min', 1e-8)
        )
    
    def set_input(self, input_data):
        """设置输入数据到指定设备"""
        self.input = input_data[0].to(self.device)
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input_data[1]]
        self.label = input_data[2].to(self.device).float()
    
    def forward(self):
        """前向传播"""
        self.get_features()
        # 处理 DDP 包装 - 如果模型是 DDP 对象，则使用 .module 访问实际模型
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.output, self.weights_max, self.weights_org = model.forward(
            self.crops, self.features
        )
        self.output = self.output.view(-1)
        
        # 计算损失
        self.loss_ral = self.criterion(self.weights_max, self.weights_org)
        self.loss_ce = self.criterion1(self.output, self.label)
        self.loss = self.loss_ral + 0.5 * self.loss_ce
    
    def get_features(self):
        """获取输入特征"""
        # 处理 DDP 包装 - 如果模型是 DDP 对象，则使用 .module 访问实际模型
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.features = model.get_features(self.input).to(self.device)
    
    def optimize_parameters(self):
        """优化参数（包括梯度累积）"""
        # 缩放损失用于梯度累积
        loss_scaled = self.loss / self.accumulation_steps
        loss_scaled.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.accumulation_count += 1
        
        # 检查是否需要更新参数
        if self.accumulation_count >= self.accumulation_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulation_count = 0
            self.update_steps += 1
            
            # 更新学习率调度器
            if self.scheduler is not None:
                self.scheduler.step()
    
    def get_loss(self):
        """获取总损失值"""
        loss = self.loss.detach().item()
        return loss
    
    def get_individual_losses(self):
        """获取各项损失值"""
        loss_ral = self.loss_ral.detach().item()
        loss_ce = self.loss_ce.detach().item()
        return loss_ral, loss_ce
    
    def train(self):
        """设置为训练模式"""
        self.model.train()
    
    def eval(self):
        """设置为评估模式"""
        self.model.eval()
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def save_networks(self, save_filename):
        """保存模型和优化器状态（仅主进程）"""
        if self.rank != 0:
            return
        
        save_path = os.path.join(self.save_dir, save_filename)
        # 处理 DDP 包装 - 如果模型是 DDP 对象，则使用 .module 获取实际模型状态
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        state_dict = {
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_steps": self.update_steps,
        }
        
        torch.save(state_dict, save_path)
        print(f"✓ 模型已保存: {save_path}")
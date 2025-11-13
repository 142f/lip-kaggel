import time
import sys
import os
import torch
import torch.nn as nn
import math
from models import build_model, get_loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainerDDP(nn.Module):
    def __init__(self, opt, rank):
        super().__init__()
        self.rank = rank
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device(f"cuda:{rank}")
        self.opt = opt
        self.model = build_model(opt.arch)

        self.step_bias = (
            0
            if not opt.fine_tune
            else int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
        )
        if opt.fine_tune:
            state_dict = torch.load(opt.pretrained_model, map_location="cpu")
            self.model.load_state_dict(state_dict["model"], strict=False)
            self.total_steps = state_dict["total_steps"]
            if self.rank == 0:
                print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")

        if opt.fix_encoder:
            for name, p in self.model.named_parameters():
                if name.split(".")[0] in ["encoder"]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            params = filter(lambda p: p.requires_grad, self.model.parameters())
        else:
            params = self.model.parameters()

        if opt.optim == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [sgd, adam, adamw]")

        # 初始化学习率调度器（如果启用）
        if opt.cosine_annealing:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=20, T_mult=1, eta_min=1e-8
            )
            self.scheduler_epoch = 0
        else:
            self.scheduler = None

        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss().to(self.device)

        # 梯度累积相关参数
        self.accumulation_steps = opt.accumulation_steps
        self.accumulation_count = 0
        # 用于跟踪实际的参数更新步骤
        self.update_steps = 0

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]
        self.label = input[2].to(self.device).float()

    def forward(self):
        self.get_features()
        self.output, self.weights_max, self.weights_org = self.model.forward(
            self.crops, self.features
        )
        self.output = self.output.view(-1)
        # 分别保存两个损失值
        self.loss_ral = self.criterion(self.weights_max, self.weights_org)
        self.loss_ce = self.criterion1(self.output, self.label)
        # 根据项目规范，CE损失项应乘以0.5的权重系数
        self.loss = self.loss_ral + 0.5 * self.loss_ce

    def get_loss(self):
        loss = self.loss.data.tolist()
        return loss[0] if isinstance(loss, type(list())) else loss

    # 添加获取单独损失值的方法
    def get_individual_losses(self):
        loss_ral = self.loss_ral.data.tolist()
        loss_ral = loss_ral[0] if isinstance(loss_ral, type(list())) else loss_ral
        loss_ce = self.loss_ce.data.tolist()
        loss_ce = loss_ce[0] if isinstance(loss_ce, type(list())) else loss_ce
        return loss_ral, loss_ce

    def optimize_parameters(self):
        # 梯度累积实现
        # 除以accumulation_steps以获得平均梯度
        loss_scaled = self.loss / self.accumulation_steps

        # 反向传播
        loss_scaled.backward()

        # 根据项目规范，添加梯度裁剪来防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.accumulation_count += 1

        # 当达到累积步数时，更新参数并清零梯度
        if self.accumulation_count == self.accumulation_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulation_count = 0
            self.update_steps += 1  # 记录实际的参数更新次数

            # 更新学习率调度器（如果启用）
            if self.scheduler is not None:
                self.scheduler.step()
        # 如果不使用梯度累积（accumulation_steps=1），也要确保调度器更新
        elif self.accumulation_steps == 1:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.update_steps += 1

            # 更新学习率调度器（如果启用）
            if self.scheduler is not None:
                self.scheduler.step()

    def get_features(self):
        self.features = self.model.get_features(self.input).to(
            self.device
        )  # shape: (batch_size

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_steps": self.update_steps,  # 保存实际更新步骤数
        }

        # 只在主进程中保存模型
        if self.rank == 0:
            torch.save(state_dict, save_path)


# 添加错误处理以更好地诊断导入问题
try:
    import ftfy
    print(f"成功导入 ftfy，版本: {ftfy.__version__}")
except ImportError as e:
    print(f"警告: 无法导入 ftfy: {e}")
    print("某些CLIP相关功能可能不可用")

try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    import torch.multiprocessing as mp
    from validate import validate
    from data import create_dataloader
    from trainer.trainer_ddp import TrainerDDP
    from options.train_options import TrainOptions
    DEPENDENCIES_AVAILABLE = True
    print("所有依赖导入成功")
except ImportError as e:
    print(f"导入模块时出错: {e}")
    DEPENDENCIES_AVAILABLE = False

if not DEPENDENCIES_AVAILABLE:
    print("错误: 必需的依赖不可用，无法继续执行")
    sys.exit(1)


# 添加日志类，用于同时输出到控制台和文件
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 使用IP地址而不是localhost
    os.environ['MASTER_PORT'] = '12355'

    # 添加一些延迟确保主进程先启动
    import time
    if rank != 0:
        time.sleep(5)

    # 等待所有进程就绪
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def get_val_opt(opt):
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    # 使用命令行参数控制验证集路径，如果没有提供则使用默认路径
    val_opt.real_list_path = opt.val_real_list_path
    val_opt.fake_list_path = opt.val_fake_list_path
    return val_opt


def format_options(opt, parser):
    """格式化选项信息，与BaseOptions.print_options方法保持一致"""
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        try:
            default = parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
        except Exception:
            pass
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    return message


def main(rank, world_size, opt):
    # 初始化分布式训练环境
    setup(rank, world_size)

    # 在主进程中创建日志
    logger = None
    if rank == 0:
        # 创建日志目录和文件（优化：放在项目根路径下的logs文件夹）
        log_dir = os.path.join("./logs", opt.name)
        os.makedirs(log_dir, exist_ok=True)
        # 优化日志文件名格式为{实验名称}_{年月日}_{时分秒}.log
        log_file = os.path.join(log_dir, f"{opt.name}_{time.strftime('%Y%m%d_%H%M%S')}.log")

        # 重定向标准输出到日志文件和控制台
        logger = Logger(log_file)
        sys.stdout = logger

        # 将训练选项写入日志文件
        train_options = TrainOptions()
        train_options.gather_options()  # 这会创建parser属性
        print(format_options(opt, train_options.parser))
        print("\n")

    # 创建训练和验证数据加载器
    train_loader = create_dataloader(opt, distributed=True)

    val_opt = get_val_opt(opt)  # 传递训练选项
    val_loader = create_dataloader(val_opt)

    if rank == 0:
        print("Length of data loader: %d" % (len(train_loader)))
        print("Length of val  loader: %d" % (len(val_loader)))

    # 创建支持DDP的训练器
    model = TrainerDDP(opt, rank)

    # 使用DDP包装模型
    model.model = DDP(model.model, device_ids=[rank])

    # 初始化最佳性能跟踪变量
    best_acc = 0.0
    best_ap = 0.0
    best_epoch = 0

    start_time = time.time()
    for epoch in range(opt.epoch):
        # 设置sampler的epoch以确保每个epoch的数据顺序不同
        train_loader.sampler.set_epoch(epoch)

        model.train()
        if rank == 0:
            print("epoch: ", epoch + model.step_bias)

        # 应用余弦退火学习率（如果启用）
        if opt.cosine_annealing:
            # 注意：PyTorch的CosineAnnealingWarmRestarts调度器会在optimizer.step()中自动更新学习率
            current_lr = model.optimizer.param_groups[0]['lr']
            if epoch % 1 == 0 and rank == 0:  # 每1个epoch打印一次学习率
                print(f"当前学习率: {current_lr:.2e}")

        for i, (img, crops, label) in enumerate(train_loader):
            model.total_steps += 1

            model.set_input((img, crops, label))
            model.forward()
            loss = model.get_loss()

            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0 and rank == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                # 获取并打印单独的损失值和总和
                loss_ral, loss_ce = model.get_individual_losses()
                total_loss = model.get_loss()
                print(
                    "Step {:6d} | loss RAL: {:8.4f} | loss CE: {:8.4f} | Total loss: {:8.4f} | Time: {:6.2f}s".format(
                        model.total_steps, loss_ral, loss_ce, total_loss, elapsed_time
                    )
                )
                start_time = time.time()

                # 添加GPU利用率监控信息
                if model.total_steps % (opt.loss_freq * 5) == 0:  # 每5次打印一次监控信息
                    if torch.cuda.is_available():
                        for gpu_id in range(torch.cuda.device_count()):
                            if gpu_id < len(opt.gpu_ids):
                                gpu_name = f"GPU {opt.gpu_ids[gpu_id]}"
                                try:
                                    gpu_util = torch.cuda.utilization(opt.gpu_ids[gpu_id]) if hasattr(torch.cuda, 'utilization') else "N/A"
                                except:
                                    gpu_util = "N/A"
                                gpu_mem = torch.cuda.memory_allocated(opt.gpu_ids[gpu_id]) / 1024**3  # GB
                                print(f"  {gpu_name}: Memory={gpu_mem:.1f}GB, Utilization={gpu_util}")

        model.eval()
        # 只在主进程中进行验证
        if rank == 0:
            ap, fpr, fnr, acc = validate(model.model.module, val_loader, opt.gpu_ids)
            print(
                "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                    epoch + model.step_bias, acc, ap, fpr, fnr
                )
            )

            # 只在验证性能超过历史最佳时才保存模型
            current_epoch = epoch + model.step_bias
            if acc > best_acc or (acc == best_acc and ap > best_ap):
                # 更新最佳性能指标
                best_acc = acc
                best_ap = ap
                best_epoch = current_epoch

                print(f" 发现新的最佳模型 (epoch {current_epoch}): acc={acc:.4f}, ap={ap:.4f}")
                model.save_networks("best_model.pth")
                # 可选：同时保存带epoch编号的模型用于记录
                model.save_networks(f"model_epoch_{current_epoch}.pth")
            else:
                print(f" 当前性能未超过最佳 (最佳: acc={best_acc:.4f}, ap={best_ap:.4f} @ epoch {best_epoch})")

    # 训练结束后打印最终的最佳性能
    if rank == 0:
        print(f"\n 训练完成！最佳模型性能:")
        print(f"   准确率 (acc): {best_acc:.4f}")
        print(f"   AP值 (ap): {best_ap:.4f}")
        print(f"   所在轮次: {best_epoch}")
        print(f"   最佳模型文件: best_model.pth")

        # 关闭日志文件
        if logger:
            logger.close()
            sys.stdout = logger.terminal  # 恢复标准输出

    # 清理分布式训练环境
    if dist.is_initialized():
        dist.destroy_process_group()


def run_training(rank, world_size, opt):
    try:
        main(rank, world_size, opt)
    except Exception as e:
        print(f"Process {rank} encountered an error: {e}")
        import traceback
        traceback.print_exc()
        # 确保在出现异常时也清理进程组
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    train_options = TrainOptions()
    opt = train_options.parse(print_options=False)  # 禁用自动打印选项

    # 获取GPU数量
    world_size = len(opt.gpu_ids) if opt.gpu_ids[0] >= 0 else 1

    # 如果只有一个GPU，直接运行
    if world_size <= 1:
        # 这里可以调用原始的训练函数
        print("请使用原始的 train.py 脚本进行单GPU训练")
    else:
        # 使用多进程启动DDP训练
        mp.spawn(run_training, args=(world_size, opt), nprocs=world_size, join=True)
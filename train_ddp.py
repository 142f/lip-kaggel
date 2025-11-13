import time
import sys
import os
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from validate import validate
from data import create_dataloader
from trainer.trainer_ddp import TrainerDDP
from options.train_options import TrainOptions


# 日志类：同时输出到控制台和日志文件，支持 with 语法
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        # 使用追加模式避免覆盖历史（根据需求也可以改回 "w"）
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        # 保证不写入空消息
        if message:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass

    # 支持 with 语法
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup(rank, world_size):
    """初始化分布式训练环境"""
    # 只有在未设置时才写环境变量，便于外部覆盖
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12355')

    # 明确指定设备ID以避免警告
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, 
                          init_method='env://')
    torch.cuda.set_device(rank)
    # 在 barrier 中明确指定 device_ids 以避免警告
    dist.barrier(device_ids=[rank])


def cleanup():
    """清理分布式训练环境"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def get_val_opt(train_opt):
    """基于训练配置生成验证配置（复用 TrainOptions）"""
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    # 使用训练时传入的验证路径（如未提供则保持默认）
    val_opt.real_list_path = getattr(train_opt, "val_real_list_path", val_opt.real_list_path)
    val_opt.fake_list_path = getattr(train_opt, "val_fake_list_path", val_opt.fake_list_path)
    return val_opt


def format_options(opt, parser):
    """格式化选项信息，与 BaseOptions.print_options 风格一致"""
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
    logger = None
    # 尝试确保 cudnn 性能优化（仅在非确定性模式下）
    if not getattr(opt, "deterministic", False):
        torch.backends.cudnn.benchmark = True

    # 设置随机种子（如果用户提供）
    if hasattr(opt, "seed") and opt.seed is not None:
        seed = int(opt.seed)
    else:
        # 如果未提供则使用时间戳作为种子（可复现性较差，但种子仍被设置）
        seed = int(time.time()) & 0xffffffff
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        # 初始化分布式环境
        setup(rank, world_size)

        # 主进程创建日志并打印配置信息
        if rank == 0:
            log_dir = os.path.join("./logs", opt.name)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{opt.name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
            logger = Logger(log_file)
            sys.stdout = logger
            # 打印训练参数到日志
            train_options = TrainOptions()
            train_options.gather_options()
            print(format_options(opt, train_options.parser))
            print("\n", flush=True)

        # 创建训练和验证 dataloader
        train_loader = create_dataloader(opt, distributed=True)
        val_opt = get_val_opt(opt)
        # 验证集通常不需要 distributed=True
        val_loader = create_dataloader(val_opt, distributed=False)

        if rank == 0:
            print(f"训练集长度: {len(train_loader)}", flush=True)
            print(f"验证集长度: {len(val_loader)}", flush=True)

        # 创建 Trainer（支持 DDP）
        model = TrainerDDP(opt, rank)

        # 将模型移动到当前设备（DDP 要求）
        model.model.cuda(rank)
        model.model = DDP(model.model, device_ids=[rank])

        best_acc = 0.0
        best_ap = 0.0
        best_epoch = 0

        # 训练循环
        for epoch in range(opt.epoch):
            # 如果使用 DistributedSampler，设置 epoch 以保证 shuffle 不同
            try:
                sampler = getattr(train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
            except Exception:
                # 只打印一次（在主进程）
                if rank == 0:
                    print("设置 sampler.set_epoch 时发生异常，已跳过该步骤。", flush=True)

            model.train()
            if rank == 0:
                print("开始 epoch: ", epoch + model.step_bias, flush=True)

            # 打印学习率（如果模型和 optimizer 可访问）
            if getattr(opt, "cosine_annealing", False):
                try:
                    current_lr = model.optimizer.param_groups[0]['lr']
                    if rank == 0:
                        print(f"当前学习率: {current_lr:.2e}", flush=True)
                except Exception:
                    pass

            start_time = time.time()
            for i, (img, crops, label) in enumerate(train_loader):
                model.total_steps += 1

                model.set_input((img, crops, label))
                model.forward()
                # 单步损失
                loss = model.get_loss()

                model.optimize_parameters()

                if model.total_steps % opt.loss_freq == 0 and rank == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    # 获取并打印单项损失和值
                    try:
                        loss_ral, loss_ce = model.get_individual_losses()
                        total_loss = model.get_loss()
                    except Exception:
                        # 兜底：若模型未实现 get_individual_losses
                        loss_ral, loss_ce = 0.0, 0.0
                        total_loss = loss if loss is not None else 0.0

                    print(
                        "Step {:6d} | loss RAL: {:8.4f} | loss CE: {:8.4f} | Total loss: {:8.4f} | Time: {:6.2f}s".format(
                            model.total_steps, loss_ral, loss_ce, total_loss, elapsed_time
                        ),
                        flush=True
                    )
                    start_time = time.time()

            model.eval()
            # 仅主进程做验证和模型保存
            if rank == 0:
                try:
                    ap, fpr, fnr, acc = validate(model.model.module, val_loader, opt.gpu_ids)
                except Exception as e:
                    print("验证阶段发生异常：", e, flush=True)
                    traceback.print_exc()
                    ap, fpr, fnr, acc = 0.0, 0.0, 0.0, 0.0

                print(
                    "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                        epoch + model.step_bias, acc, ap, fpr, fnr
                    ),
                    flush=True
                )

                current_epoch = epoch + model.step_bias
                if acc > best_acc or (acc == best_acc and ap > best_ap):
                    best_acc = acc
                    best_ap = ap
                    best_epoch = current_epoch

                    print(f"发现新的最佳模型 (epoch {current_epoch}): acc={acc:.4f}, ap={ap:.4f}", flush=True)
                    # 保存最佳模型（相对路径）
                    model.save_networks("best_model.pth")
                    model.save_networks(f"model_epoch_{current_epoch}.pth")
                else:
                    print(f"当前性能未超过最佳 (最佳: acc={best_acc:.4f}, ap={best_ap:.4f} @ epoch {best_epoch})", flush=True)

            # 每个 epoch 后尝试释放显存碎片（可选）
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # 训练结束主进程打印总结
        if rank == 0:
            print("\n训练完成！最佳模型性能:", flush=True)
            print(f" 准确率 (acc): {best_acc:.4f}", flush=True)
            print(f" AP值 (ap): {best_ap:.4f}", flush=True)
            print(f" 所在轮次: {best_epoch}", flush=True)
            print(f" 最佳模型文件: best_model.pth", flush=True)

    except Exception as e:
        # 主进程打印完整错误，其他进程打印简短信息
        if rank == 0:
            print("训练过程中出现异常：", e, flush=True)
            traceback.print_exc()
        else:
            print(f"进程 {rank} 遇到错误: {e}", flush=True)
        # 确保即使出错也会执行 finally 释放资源
    finally:
        # 恢复标准输出并关闭日志（主进程）
        if rank == 0 and logger is not None:
            try:
                logger.close()
            except Exception:
                pass
            try:
                sys.stdout = logger.terminal
            except Exception:
                pass

        # 清理分布式进程组
        cleanup()


def run_training(rank, world_size, opt):
    # 每个子进程都调用 main
    main(rank, world_size, opt)


if __name__ == "__main__":
    train_options = TrainOptions()
    opt = train_options.parse(print_options=False)

    # 计算 world_size（兼容 gpu_ids = [-1] 或空）
    try:
        gpu_ids = getattr(opt, "gpu_ids", None)
        if gpu_ids is None or len(gpu_ids) == 0 or gpu_ids[0] < 0:
            world_size = 1
        else:
            world_size = len(gpu_ids)
    except Exception:
        world_size = 1

    if world_size <= 1:
        print("当前检测到单 GPU 或 CPU 环境。若需单卡训练，请使用单卡训练脚本（例如原始 train.py）。", flush=True)
    else:
        # 为确保 spawn 启动方式一致性，显式设置
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果已经设置过则忽略
            pass
        mp.spawn(run_training, args=(world_size, opt), nprocs=world_size, join=True)
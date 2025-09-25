# trainer/trainer.py
import os
import torch
import torch.nn as nn
from models import get_loss # 不再需要 build_model


class Trainer(nn.Module):
    # --- 1. 修改 __init__ 方法，接收一个外部 model 对象 ---
    def __init__(self, opt, model):
        super().__init__()
        self.opt = opt
        self.model = model # 直接使用外部传入的模型
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # self.device 始终是主设备
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )

        # --- 2. 将模型加载逻辑移到外部 (train.py)，但保留恢复 step 的逻辑 ---
        self.step_bias = (
            0
            if not opt.fine_tune
            else int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
        )
        if opt.fine_tune:
            # 假设模型已经在外部加载了权重，这里只恢复 total_steps
            # 注意：更稳健的做法是从 state_dict 中读取 total_steps
            try:
                state_dict = torch.load(opt.pretrained_model, map_location="cpu")
                self.total_steps = state_dict.get("total_steps", 0) # 使用 .get 避免 key 不存在时报错
                print(f"Trainer total_steps restored to {self.total_steps}")
            except Exception as e:
                print(f"Could not restore total_steps from checkpoint: {e}")


        # --- 3. 优化器创建逻辑保持，它会正确处理 DataParallel 包装过的模型 ---
        # 如果模型被 DataParallel 包装, self.model.parameters() 会返回原始模型的参数
        if opt.fix_encoder:
            for name, p in self.model.named_parameters():
                if "encoder" in name: # 更稳健的检查方式
                    p.requires_grad = False
            params = filter(lambda p: p.requires_grad, self.model.parameters())
        else:
            params = self.model.parameters()

        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        # --- 4. 损失函数和后续代码 ---
        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss()

        # 注意：模型已经在外部被 .to(device) 了，这里不再需要移动

    def adjust_learning_rate(self, min_lr=1e-8):
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < min_lr:
                return False
            param_group["lr"] /= 10.0
        return True

    # set_input 方法是正确的，因为它把所有数据都放到了主设备上
    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]
        self.label = input[2].to(self.device).float()

    def forward(self):
        # 注意: get_features 也需要在 DataParallel 模型上调用
        self.get_features()
        # model.forward 会被 DataParallel 自动分发到各个 GPU
        self.output, self.weights_max, self.weights_org = self.model.forward(
            self.crops, self.features
        )
        self.output = self.output.view(-1)
        # loss 计算在主设备上进行
        self.loss = self.criterion(
            self.weights_max, self.weights_org
        ) + self.criterion1(self.output, self.label)

    def get_loss(self):
        # DataParallel 会将多个 GPU 的 loss 聚合到主设备上，所以 self.loss 是一个标量
        return self.loss.item() # 使用 .item() 获取纯 Python 数字

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss.backward() # loss.backward() 会自动处理多 GPU 的梯度聚合
        self.optimizer.step()

    def get_features(self):
        # self.model.get_features 会被 DataParallel 代理到原始模型上
        self.features = self.model.get_features(self.input)
        # 注意：这里的 .to(self.device) 可能多余，因为 get_features 的输出应该已经在 GPU 上
        # 但保留它也不会出错

    def train(self): # 添加一个 train 方法，方便主脚本调用
        self.model.train()

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    # --- 5. 修改 save_networks 方法以支持 DataParallel ---
    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)

        # 判断模型是否被 DataParallel 包装
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        # 序列化模型和优化器等信息
        state_dict = {
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(state_dict, save_path)
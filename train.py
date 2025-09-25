import torch
from validate import validate
from data import create_dataloader
from models import build_model # 在这里导入 build_model
from trainer.trainer import Trainer
from options.train_options import TrainOptions


def get_val_opt(main_opt): # 让 val_opt 继承主 opt 的一些设置
    val_opt = TrainOptions().parse(print_options=False)
    # 继承一些通用设置
    val_opt.arch = main_opt.arch
    val_opt.gpu_ids = main_opt.gpu_ids
    # 设置 val 特有的参数
    val_opt.isTrain = False
    val_opt.data_label = "val"
    val_opt.real_list_path = r"/kaggle/input/val-ckpt/val/val/0_real"
    val_opt.fake_list_path = r"/kaggle/input/val-ckpt/val/val/1_fake"
    return val_opt


if __name__ == "__main__":
    # 1. 初始化配置和设备
    opt = TrainOptions().parse()
    val_opt = get_val_opt(opt)

    # 2. 创建模型、加载权重、并根据GPU数量进行适配
    if opt.gpu_ids:
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
        torch.cuda.set_device(device)
        print(f"Using GPUs: {opt.gpu_ids}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # 创建原始模型实例
    model_instance = build_model(opt.arch)

    # 加载预训练权重 (如果有)
    if opt.fine_tune:
        state_dict = torch.load(opt.pretrained_model, map_location="cpu")
        model_instance.load_state_dict(state_dict["model"])
        print(f"Model weights loaded from {opt.pretrained_model.split('/')[-1]}")

    # 根据 GPU 数量决定是否使用 DataParallel
    if len(opt.gpu_ids) > 1:
        print(f"Activating DataParallel for {len(opt.gpu_ids)} GPUs.")
        model_instance = torch.nn.DataParallel(model_instance, device_ids=opt.gpu_ids)

    # 将模型移动到主设备
    model_instance.to(device)

    # 3. 将准备好的模型传递给 Trainer
    trainer = Trainer(opt, model_instance)

    # 4. 数据加载
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    print("Length of data loader: %d" % (len(data_loader)))
    print("Length of val  loader: %d" % (len(val_loader)))

    # 5. 训练循环
    for epoch in range(trainer.step_bias, opt.epoch):
        trainer.train()
        print("epoch: ", epoch)

        for i, (img, crops, label) in enumerate(data_loader):
            trainer.total_steps += 1
            trainer.set_input((img, crops, label))
            trainer.forward()
            trainer.optimize_parameters()

            if trainer.total_steps % opt.loss_freq == 0:
                print(
                    "Train loss: {:.4f}\tstep: {}".format(
                        trainer.get_loss(), trainer.total_steps
                    )
                )

        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" % epoch)
            trainer.save_networks(f"model_epoch_{epoch}.pth")

        trainer.eval()
        ap, fpr, fnr, acc = validate(trainer.model, val_loader, opt.gpu_ids)
        print(
            "(Val @ epoch {}) acc: {:.4f} ap: {:.4f} fpr: {:.4f} fnr: {:.4f}".format(
                epoch, acc, ap, fpr, fnr
            )
        )
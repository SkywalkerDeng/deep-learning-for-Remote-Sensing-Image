import torch
import time
# 导入 UNet 和其他可能需要的模型 (使用相对路径导入模型更安全)
# 如果 model 目录与 tools 在同一级 (都在 FCN_frame下)，需要 ..
# 但因为我们将 FCN_frame 加入了 PYTHONPATH，尝试绝对导入
try:
    from model.unet import UNet
    from model import fcn_resnet50, VGG16UNet # 假设这些也在 model 包下
    from model.deeplabv3_model import deeplabv3_resnet50 # 假设这个也在 model 包下
except ImportError:
    # 如果绝对导入失败，尝试相对导入 (假设 model 和 tools 是 FCN_frame 下的同级目录)
    print("绝对导入模型失败，尝试相对导入...")
    from ..model.unet import UNet
    from ..model import fcn_resnet50, VGG16UNet
    from ..model.deeplabv3_model import deeplabv3_resnet50


import torch.optim as optim
from torch import nn
# from torch.nn.functional import cross_entropy as criterion # criterion 定义在下面
import numpy as np
# 使用相对导入 utils (因为 utils.py 和 engine.py 在同一个 tools 目录下)
from .utils import compute_miou, compute_iou_per_class
import datetime
import sys # 引入 sys 用于进度条
from tqdm import tqdm # 引入 tqdm 用于进度条

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


# 损失函数定义 (来自你之前提供的代码)
def criterion_(inputs: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        # PyTorch CrossEntropyLoss 默认处理 LongTensor
        losses[name] = nn.functional.cross_entropy(x, target.long(), ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    # 注意：这里的 aux loss 权重是 0.5
    return losses['out'] + 0.5 * losses['aux']

# 修正后的 create_model 函数
def create_model(model_name: str, num_classes: int, aux: bool = False, in_channels: int = 3):
    """
    Creates a segmentation model.

    Args:
        model_name (str): Name of the model (e.g., 'unet', 'fcn_resnet50').
        num_classes (int): Number of output classes.
        aux (bool): Whether models that support auxiliary loss should enable it.
        in_channels (int): Number of input channels for the model. Default is 3.

    Returns:
        torch.nn.Module: The created model.
    """
    model = None # Initialize model to None
    if model_name == "fcn_resnet50":
        # Assuming fcn_resnet50 expects aux and num_classes. Does it need in_channels? Needs verification.
        model = fcn_resnet50(aux=aux, num_classes=num_classes)
        print(f"Created model: {model_name}")
    elif model_name == "unet":
        # *** 修正了 UNet 的调用 ***
        # 传递了 in_channels 和 num_classes, 移除了 pretrain_backbone
        # 假设 unet.py 中的类参数是 n_channels 和 n_classes
        model = UNet(n_channels=in_channels, n_classes=num_classes)
        print(f"Created model: {model_name} with in_channels={in_channels}, num_classes={num_classes}")
    elif model_name == "vgg16unet":
        # 假设 VGG16UNet 也需要 in_channels 和 num_classes，并移除了 pretrain_backbone
        # !! 需要确认 VGG16UNet 的实际参数 !!
        model = VGG16UNet(in_channels=in_channels, num_classes=num_classes) # 假设参数名
        print(f"Created model: {model_name}")
    elif model_name == "deeplabv3_resnet50":
        # 假设 deeplabv3_resnet50 需要 aux, num_classes。是否需要 in_channels?
        # 移除了 pretrain_backbone
        model = deeplabv3_resnet50(aux=aux, num_classes=num_classes) # 可能需要添加 in_channels
        print(f"Created model: {model_name}")
    else:
        # 如果没有匹配的模型名称，抛出错误
        raise ValueError(f"Model name '{model_name}' is not recognized in create_model function.")

    return model

# create_optimizer 函数 (来自你之前提供的代码)
def create_optimizer(opt_name, lr_, model):
    if opt_name=="adamw":
      optimizer = optim.AdamW(model.parameters(), lr=lr_) # 使用 AdamW 通常更好
    elif opt_name=="sgd": # 添加一个 SGD 选项示例
         optimizer = optim.SGD(model.parameters(), lr=lr_, momentum=0.9, weight_decay=0.0001)
    else:
         print(f"Warning: Optimizer '{opt_name}' not explicitly handled, using AdamW as default.")
         optimizer = optim.AdamW(model.parameters(), lr=lr_)
    return optimizer


# train 函数 (来自你之前提供的代码, 稍作调整和注释)
def train(num_epochs, dataloader, val_dataloader, optimizer, model, aux, weight_path, device, batch_size, num_classes, lr_scheduler):
    epoch_times = []
    train_losses = []
    best_mIoU = 0.0 # 通常用 mIoU 作为最佳模型的标准
    results_file = f"/kaggle/working/training_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    print(f"Training results will be saved to: {results_file}")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train() # 设置为训练模式
        running_loss = 0.0
        # 使用 tqdm 显示训练进度条
        train_bar = tqdm(dataloader, file=sys.stdout, desc=f"Epoch [{epoch+1}/{num_epochs}] Training")
        for i, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device).long() # 确保 labels 是 LongTensor

            # --- 检查数据加载后的形状 (仅调试用，可注释掉) ---
            # if i == 0 and epoch == 0:
            #     print(f"\nFirst batch - Images shape: {images.shape}, dtype: {images.dtype}")
            #     print(f"First batch - Labels shape: {labels.shape}, dtype: {labels.dtype}")
            #     print(f"First batch - Labels min: {labels.min()}, max: {labels.max()}")
            # --- 调试结束 ---

            optimizer.zero_grad()
            outputs = model(images) # 前向传播

            # 计算损失 (根据是否有辅助输出来选择)
            if aux and isinstance(outputs, dict) and 'aux' in outputs:
                loss = criterion_(outputs, labels)
            elif isinstance(outputs, dict) and 'out' in outputs:
                 loss = nn.functional.cross_entropy(outputs['out'], labels, ignore_index=255)
            else: # 假设直接输出 logits
                 loss = nn.functional.cross_entropy(outputs, labels, ignore_index=255)

            loss.backward()
            optimizer.step()
            lr_scheduler.step() # 更新学习率

            running_loss += loss.item()
            # 更新进度条显示
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")


        # --- 每个 epoch 结束后进行验证 ---
        model.eval() # 设置为评估模式
        all_preds_np = []
        all_labels_np = []
        val_loss = 0.0
        val_steps = 0

        val_bar = tqdm(val_dataloader, file=sys.stdout, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation")
        with torch.no_grad():
            for images_val, labels_val in val_bar:
                images_val, labels_val = images_val.to(device), labels_val.to(device).long()
                outputs_val = model(images_val)

                # 计算验证损失
                if aux and isinstance(outputs_val, dict) and 'aux' in outputs_val:
                     loss_val = criterion_(outputs_val, labels_val)
                elif isinstance(outputs_val, dict) and 'out' in outputs_val:
                     loss_val = nn.functional.cross_entropy(outputs_val['out'], labels_val, ignore_index=255)
                     logits_val = outputs_val['out']
                else:
                     loss_val = nn.functional.cross_entropy(outputs_val, labels_val, ignore_index=255)
                     logits_val = outputs_val

                val_loss += loss_val.item()
                val_steps += 1

                # 获取预测结果用于计算指标
                preds_val = torch.argmax(logits_val, dim=1)
                all_preds_np.append(preds_val.cpu().numpy())
                all_labels_np.append(labels_val.cpu().numpy())

        # --- 计算整个验证集的指标 ---
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        avg_train_loss_epoch = running_loss / len(dataloader) if len(dataloader) > 0 else 0

        if not all_preds_np: # 如果验证集为空
             print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss_epoch:.4f} - Validation set is empty.")
             mIoU = 0 # 或者其他标记值
             accuracy=0; m_precision=0; m_F1=0; ua_str="N/A"; f1_str="N/A"; iou_str="N/A" # 赋默认值
        else:
             all_preds_np = np.concatenate(all_preds_np).flatten() # 展平用于计算指标
             all_labels_np = np.concatenate(all_labels_np).flatten()

             # 过滤掉 ignore_index (255)
             valid_pixels = all_labels_np != 255
             all_preds_np = all_preds_np[valid_pixels]
             all_labels_np = all_labels_np[valid_pixels]

             if len(all_labels_np) == 0: # 如果过滤后没有有效像素
                 print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss_epoch:.4f} - No valid pixels found in validation labels after filtering.")
                 mIoU = 0; accuracy=0; m_precision=0; m_F1=0; ua_str="N/A"; f1_str="N/A"; iou_str="N/A"
             else:
                 # 计算指标
                 accuracy = accuracy_score(all_labels_np, all_preds_np)
                 # 使用 zero_division=0 避免当某个类别从未被预测或从未出现时 F1/Precision 报错
                 m_precision, _, m_F1, _ = precision_recall_fscore_support(all_labels_np, all_preds_np, average='macro', zero_division=0)
                 precision, _, F1, _= precision_recall_fscore_support(all_labels_np, all_preds_np, average=None, zero_division=0)
                 IoU = compute_iou_per_class(all_preds_np, all_labels_np, num_classes=num_classes)
                 mIoU = compute_miou(all_preds_np, all_labels_np, num_classes=num_classes) # compute_miou 内部处理了 NaN

                 ua_str = ', '.join([f'{p:.4f}' for p in precision])
                 f1_str = ', '.join([f'{f1:.4f}' for f1 in F1])
                 iou_str = ', '.join([f'{iou:.4f}' if not np.isnan(iou) else 'nan' for iou in IoU])


        # --- 打印和保存结果 ---
        epoch_summary = (
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {avg_train_loss_epoch:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"OA: {accuracy:.4f}, mIoU: {mIoU:.4f}, mF1: {m_F1:.4f}, mPrecision: {m_precision:.4f}\n"
            f"  Class IoU: [{iou_str}]\n"
            f"  Class F1: [{f1_str}]\n"
            f"  Class Precision: [{ua_str}]"
        )
        print(epoch_summary)

        with open(results_file, "a") as f:
            f.write(epoch_summary + "\n")

        # 保存最佳模型 (基于 mIoU)
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            print(f"  New best mIoU: {best_mIoU:.4f}. Saving model to {weight_path}...")
            torch.save(model.state_dict(), weight_path)

        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        time_info = f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s"
        print(time_info)
        with open(results_file, "a") as f:
             f.write(time_info + "\n\n")


    # --- 训练结束 ---
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    final_summary = f"\nTraining finished. Average time per epoch: {avg_epoch_time:.2f}s. Best mIoU: {best_mIoU:.4f}"
    print(final_summary)
    with open(results_file, "a") as f:
        f.write(final_summary + "\n")

    return model


# create_lr_scheduler 函数 (来自你之前提供的代码)
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            # Polynomial decay.
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


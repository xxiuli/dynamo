#temsorboard_utils.py
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
from datetime import datetime
import os

def create_writer(log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_log_dir = os.path.join(log_dir, timestamp)
    os.makedirs(final_log_dir, exist_ok=True)
    return SummaryWriter(final_log_dir)

def plot_confusion_matrix_to_tensorboard(preds, labels, class_names, writer, epoch, tag="ConfusionMatrix"):
    """
    将混淆矩阵绘图输出到 TensorBoard
    - preds: 模型预测 (list 或 ndarray)
    - labels: 真实标签 (list 或 ndarray)
    - class_names: 类别名称 (list[str])
    - writer: TensorBoard writer 实例
    - epoch: 当前 epoch
    - tag: 可选，tensorboard 中的命名标签
    """
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

     # 绘图
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"{tag} - Epoch {epoch}")
    plt.tight_layout()

    # 转换为 TensorBoard 可接受的图像格式
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)

    # 写入 TensorBoard
    writer.add_image(tag, image, global_step=epoch, dataformats='HWC')
    plt.close(fig)

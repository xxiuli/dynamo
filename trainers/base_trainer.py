# BaseTrainer.py
import os
from transformers import get_scheduler
from utils.tensorboard_utils import create_writer
from utils.train_utils import get_optimizer,EarlyStop, save_model
from tqdm.auto import tqdm
from abc import ABC, abstractmethod  # 如果要强制继承类实现 evaluate

class BaseTrainer(ABC):
    def __init__(self, model, config, device, tokenizer):
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

        # output paths
        try:
            self.save_dir = self.config['output']['save_dir']
            os.makedirs(self.save_dir, exist_ok=True)
            self.log_dir = self.config['output'].get('log_dir', 'runs/default')
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            raise OSError(f"[ERROR] Failed to create output/log directories: {e}")

        try:
            self.writer = create_writer(log_dir=self.log_dir)
        except Exception as e:
            raise IOError(f"[ERROR] Failed to create TensorBoard writer at {self.log_dir}: {e}")

        # class_names(labels)
        self.class_names = self.config.get("class_names", [str(i) for i in range(self.config['num_labels'])])

        # optimizer and scheduler
        train_config = self.config['train']
        self.optimizer = get_optimizer(
            train_config['optimizer'], 
            filter(lambda p: p.requires_grad, model.parameters()), 
            float(train_config['learning_rate'])
            )
        
        total_steps = train_config['num_epochs'] * train_config['steps_per_epoch']

        self.scheduler = get_scheduler(
                name=train_config['lr_scheduler'],
                optimizer=self.optimizer,
                num_warmup_steps=int(train_config.get('warmup_ratio', 0) * total_steps),
                num_training_steps=total_steps
            )
        
        # Early Stopping
        self.early_stopper = None
        if self.config['early_stopping']['enabled']:
            mode = self.config['early_stopping']['mode']
            self.early_stopper = EarlyStop(patience=self.config['early_stopping']['patience'], mode=mode)
        
        #save mode
        self.save_best_only = self.config['output'].get('save_best_only', True)
        self.best_val_loss = float('inf')

        self.monitor_metric = self.config['output'].get('monitor_metric', 'val_loss')
        self.monitor_mode = self.config['early_stopping'].get('mode', 'min')
        self.best_metric = float('-inf') if self.monitor_mode == 'max' else float('inf')
    
    def train(self, train_loader, val_loader):
        debug_config = self.config.get("debug", {})
        skip_training = debug_config.get("skip_training", False)
        skip_evaluation = debug_config.get("skip_evaluation", False)

        # 当本地DEBUG跑通脚本的时候
        # ====== [DEBUG] 跳过训练，直接测试 evaluate 和 save 流程 ======
        if skip_training:
            print("[DEBUG] Skipping training loop. Directly testing evaluation & saving logic.")
            if not skip_evaluation:
                val_metrics = self.evaluate(val_loader, epoch=0)
                print(f"[DEBUG] Evaluation metrics: {val_metrics}")
            else:
                print("[DEBUG] Skipped evaluation.")
            save_model(self, final=True)
            self.writer.close()
            return

        num_epochs = self.config['train']['num_epochs']

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            # for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', dynamic_ncols=True)):
            for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
                batch = {k :v.to(self.device) for k, v in batch.items()}
                
                # 否则
                # 通用 forward step
                outputs, loss = self._forward_step(batch)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()

                global_step = epoch * len(train_loader) + step
                self.writer.add_scalar('Loss/Batch_Train', loss.item(), global_step)
            
            avg_train_loss = total_loss / len(train_loader)

            if not skip_evaluation:

                val_metrics = self.evaluate(val_loader, epoch)
                val_loss = val_metrics.get('val_loss', None)
                current_metric = val_metrics.get(self.monitor_metric)

                if current_metric is None:
                    raise ValueError(f"[ERROR] Monitor metric '{self.monitor_metric}' not found in evaluate() return.")

                # 打印所有 metrics
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, ", end="")
                for k, v in val_metrics.items():
                    print(f"{k}={v:.4f} ", end="")
                print()

                if self.early_stopper:
                    self.early_stopper(current_metric)
                    if self.early_stopper.early_stop:
                        print(f"[INFO] Early stopping triggered after epoch {epoch+1}.")
                        break

                is_better = (
                    current_metric > self.best_metric if self.monitor_mode == 'max'
                    else current_metric < self.best_metric
                )

                if not self.save_best_only or is_better:
                    self.best_metric = current_metric
                    save_model(self, epoch=epoch)

                self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('LearningRate', self.scheduler.get_last_lr()[0], epoch)
            else:
                print(f"[DEBUG] Skipped evaluation at epoch {epoch+1}.")
            
        save_model(self, final=True)
        
        self.writer.close()

    @abstractmethod
    def evaluate(self, val_loader, epoch):
        pass
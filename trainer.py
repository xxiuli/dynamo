import os
import yaml
from utils.tensorboard_utils import create_writer, plot_confusion_matrix_to_tensorboard
from utils.train_utils import get_optimizer
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import get_scheduler

class EarlyStop:
    def __init__(self, patience=3, mode='min'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode =='min' and score >= self.best_score) or (self.mode == 'max' and score <= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class SingleTaskTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # criterion
        # self.criterion = CrossEntropyLoss()

        # output paths
        self.save_dir = config['output']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_dir = config['output'].get('log_dir', 'runs/default')
        os.makedirs(self.log_dir, exist_ok=True)

        try:
            self.writer = create_writer(log_dir=self.log_dir)
        except Exception as e:
            raise IOError(f"[ERROR] Failed to create TensorBoard writer at {self.log_dir}: {e}")

        # class_names(labels)
        self.class_names = config.get("class_names", [str(i) for i in range(config['num_labels'])])

        # optimizer and scheduler
        train_config = config['train']
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
        if config['early_stopping']['enabled']:
            mode = 'min' if config['early_stopping']['monitor'] == 'val_loss' else 'max'
            self.early_stopper = EarlyStop(patience=config['early_stopping']['patience'], mode=mode)
        
        #save mode
        self.save_best_only = config['output'].get('save_best_only', True)
        self.best_val_loss = float('inf')

    def train(self, train_loader, val_loader, tokenizer=None):
        num_epochs = self.config['train']['num_epochs']

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
                batch = {k :v.to(self.device) for k, v in batch.items()}
                output = self.model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'],
                    labels=batch['label']
                    )
                
                loss = output.loss
                # loss = self.criterion(output.logits, batch['label'])

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()

                global_step = epoch * len(train_loader) + step
                self.writer.add_scalar('Loss/Batch_Train', loss.item(), global_step)
            
            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_acc = self.evaluate(val_loader, epoch)

            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            if self.early_stopper:
                self.early_stopper(val_loss)
                if self.early_stopper.early_stop:
                    print(f"[INFO] Early stopping triggered after epoch {epoch+1}.")
                    break
            
            if not self.save_best_only or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model()

            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('LearningRate', self.scheduler.get_last_lr()[0], epoch)
        
        self.save_model(final=True, tokenizer=tokenizer)
        self.writer.close()

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        preds, labels = [], [] 

        try:
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    output = self.model(
                        input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'],
                        labels=batch['label']
                        )
                    loss =  output.loss
                    # loss = self.criterion(output.logits, batch['label'])
                    total_loss += loss.item()
                    pred = output.logits.argmax(dim=-1).cpu().numpy()
                    preds.extend(pred)
                    labels.extend(batch['label'].cpu().numpy())
                
                acc = accuracy_score(labels, preds)
                ave_loss = total_loss / len(val_loader)
                plot_confusion_matrix_to_tensorboard(preds, labels, self.class_names, self.writer, epoch)
                return ave_loss, acc
        except Exception as e:
            print(f"[ERROR] Evaluation failed at epoch {epoch+1}: {e}")
            return float("inf"), 0.0
        
    def save_model(self, final=False, tokenizer=None):
        try:
            path = os.path.join(self.save_dir, 'final' if final else '')
            os.makedirs(path, exist_ok=True)

            # 保存 adapter
            self.model.save_pretrained(path) 

            # 保存解码器
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            # torch.save(self.model.classifier.state_dict(), os.path.join(path, "classifier_head.pt"))
            
            # 保存 base model（只保存一次）
            if final and hasattr(self.model, 'base_model'):
                base_path = os.path.join(self.save_dir, 'base')
                self.model.base_model.save_pretrained(base_path)

            # 保存训练此模型的CONFIG
            with open(os.path.join(self.save_dir, 'final', 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f)

            print(f"[INFO] Model saved to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
# trainer_cls_token.py ->conll03
import torch
from sklearn.metrics import classification_report
from trainers.base_trainer import BaseTrainer
from sklearn.metrics import f1_score
from utils.tensorboard_utils import plot_confusion_matrix_to_tensorboard

class TokenClassificationTrainer(BaseTrainer):
    def __init__(self, model, config, device, tokenizer):
        super().__init__(model, config, device, tokenizer)
        try:
            self.label2id = config['label2id']
            self.id2label = {v: k for k, v in self.label2id.items()}
        except KeyError as e:
            raise ValueError(f"[ERROR] label2id not found in config: {e}")

    def _forward_step(self, batch):
        try:
            outputs = self.model(**batch)
            return outputs, outputs["loss"]
        except Exception as e:
            raise RuntimeError(f"[ERROR] Model forward pass failed: {e}")

    def evaluate(self, val_loader, epoch):
        debug_config = self.config.get("debug", {})
        skip_evaluation = debug_config.get("skip_evaluation", False)
        limit_batches = debug_config.get("limit_batches", None)

        if skip_evaluation:
            print("[DEBUG] Skipping evaluation step as per config.")
            return {
                "val_loss": 0.0,
                "val_f1": 0.0
            }

        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        try:
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    try:
                        # 如果本地调试，limit_batches=2， 那么就跑2个EPOACH就停了
                        # if limit_batches is not None and i >= limit_batches:
                        #     break

                        if not isinstance(batch['labels'], torch.Tensor):
                            batch['labels'] = torch.tensor(batch['labels'], dtype=torch.long)
                        else:
                            batch['labels'] = batch['labels'].to(torch.long)

                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        total_loss += loss.item()

                        predictions = torch.argmax(outputs.logits, dim=-1)
                        labels = batch['labels']

                        for pred, label in zip(predictions, labels):
                            pred = pred.cpu().numpy().tolist()
                            label = label.cpu().numpy().tolist()
                            for p, l in zip(pred, label):
                                if l != -100:
                                    all_preds.append(p)
                                    all_labels.append(l)
                    except Exception as e:
                        batch_keys = list(batch.keys()) if isinstance(batch, dict) else "Unavailable"
                        print(f"[WARNING] Skipped batch {i} due to error: {e}. Batch keys: {batch_keys}")

            # try: 
            #     if isinstance(all_preds, list):
            #         all_preds = torch.cat(all_preds).cpu().numpy()
            #     if isinstance(all_labels, list):
            #         all_labels = torch.cat(all_labels).cpu().numpy()

            #     assert len(all_preds) == len(all_labels), f"preds: {len(all_preds)}, labels: {len(all_labels)}"
            # except Exception as e:
            #     print(f"[ERROR] Failed to concat predictions: {e}")
            #     return {"val_loss": float("inf"), "val_acc": 0.0}

            if len(all_preds) == 0:
                print(f"[WARNING] No valid predictions to evaluate at epoch {epoch}")
                return {"val_loss": float("inf"), "val_acc": 0.0}

            val_loss = total_loss / max(1, len(val_loader)) # 防止被0除

            try: 
                report = classification_report(
                    all_labels, 
                    all_preds,
                    labels=sorted(self.id2label.keys()),  # ✅ 显式告诉它有哪些类
                    target_names=[self.id2label[i] for i in sorted(self.id2label.keys())],
                    digits=4
                )

                macro_f1 = f1_score(all_labels, all_preds, average='macro')

                self.writer.add_scalar("F1/Macro", macro_f1, epoch)
                self.writer.add_text("Classification Report", report, epoch)
            except Exception as e:
                print(f"[ERROR] Failed to compute classification report: {e}")
                report = "Report unavailable"
                macro_f1 = 0.0

            # 返回 dict，让 BaseTrainer 可以通用读取任何 metric
            return {
                "val_loss": val_loss,
                "val_f1": macro_f1
            }
        
        except Exception as e:
            print(f"[ERROR] Evaluation failed at epoch {epoch}: {e}")
            return {
                "val_loss": float("inf"),
                "val_f1": 0.0
            }
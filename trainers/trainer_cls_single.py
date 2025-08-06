# trainers/trainer_cls_single.py  -> sst2, agnews, mnli, qqp
import torch
from sklearn.metrics import accuracy_score, classification_report
from trainers.base_trainer import BaseTrainer
from utils.tensorboard_utils import plot_confusion_matrix_to_tensorboard

class SingleClassificationTrainer(BaseTrainer):
    def __init__(self, model, config, device, tokenizer):
        super().__init__(model, config, device, tokenizer)
        self.best_acc = 0.0

    def _forward_step(self, batch):
        outputs = self.model(**batch) ## trigger all functions of MODEL（ROBERTA+HEAD) to be executed
        return outputs, outputs["loss"]

    def evaluate(self, val_loader, epoch):

        debug_config = self.config.get("debug", {})
        skip_evaluation = debug_config.get("skip_evaluation", False)
        limit_batches = debug_config.get("limit_batches", None)

        if skip_evaluation:
            print("[DEBUG] Skipping evaluation step as per config.")
            return {
                "val_loss": 0.0,
                "val_acc": 0.0
            }

        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                try:
                    # this is for local debug，limit_batches=2， stop after 2 epoch
                    # if limit_batches is not Null and i >= limit_batches:
                    #     break

                    # print(f"[DEBUG] Batch {i} raw labels: {batch['labels'][:5]}")
                    # print(f"[DEBUG] Dtype: {batch['labels'].dtype}, shape: {batch['labels'].shape}, type: {type(batch['labels'])}")

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

                    all_preds.append(predictions.cpu()) 
                    all_labels.append(labels.cpu())
                except Exception as e:
                    batch_keys = list(batch.keys()) if isinstance(batch, dict) else "Unavailable"
                    print(f"[WARNING] Skipped batch {i} due to error: {e}. Batch keys: {batch_keys}")

        try: 
            if isinstance(all_preds, list):
                all_preds = torch.cat(all_preds).cpu().numpy()
            if isinstance(all_labels, list):
                all_labels = torch.cat(all_labels).cpu().numpy()

            assert len(all_preds) == len(all_labels), f"preds: {len(all_preds)}, labels: {len(all_labels)}"
        except Exception as e:
            print(f"[ERROR] Failed to concat predictions: {e}")
            return {"val_loss": float("inf"), "val_acc": 0.0}

        if len(all_preds) == 0:
            print(f"[WARNING] No valid predictions to evaluate at epoch {epoch}")
            return {"val_loss": float("inf"), "val_acc": 0.0}
        
        val_loss = total_loss / max(1, len(val_loader))  # preventing 0/

        try:
            report = classification_report(all_labels, all_preds, digits=4)
            acc = accuracy_score(all_labels, all_preds)
            self.writer.add_scalar("Acc/Val", acc, epoch)
            self.writer.add_text("Classification Report", report, epoch)
        except Exception as e:
            print(f"[ERROR] Failed to compute metrics: {e}")
            report = "Report unavailable"
            acc = 0.0

        # visualize 
        try:
            # save confusion matrix only if best acc
            if acc > self.best_acc:
                self.best_acc = acc
                self.best_epoch = epoch
                plot_confusion_matrix_to_tensorboard(
                    preds=all_preds,
                    labels=all_labels,
                    class_names=self.class_names,
                    writer=self.writer,
                    epoch=epoch  # best epoch
                )
        except Exception as e:
            print(f"[ERROR] Failed to plot confusion matrix in epoch: {epoch}: {e}")

        return {
            "val_loss": val_loss,
            "val_acc": acc
        }
# train_qa.py ->squad
from sklearn.metrics import accuracy_score
from trainers.base_trainer import BaseTrainer
import torch

class QuestionAnsweringTrainer(BaseTrainer):
    def __init__(self, model, config, device, tokenizer):
        super().__init__(model, config, device, tokenizer)

    def _forward_step(self, batch):
        outputs = self.model(**batch) ## 这里会触发所有的MODEL（ROBERTA+HEAD)里的函数执行
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
        total_loss = 0.0
        all_start_preds, all_end_preds = [], []
        all_start_labels, all_end_labels = [], []

        try:
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                     # 如果本地调试，limit_batches=2， 那么就跑2个EPOACH就停了
                    if limit_batches is not None and i >= limit_batches:
                        break
                    try:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        print(f"[DEBUG] Epoch {epoch}, Batch {i}, Labels: {batch['labels']}")

                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            start_positions=batch["start_positions"],
                            end_positions=batch["end_positions"]
                        )

                        loss = outputs.loss
                        print(f"[DEBUG] Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

                        total_loss += loss.item()

                        start_preds = torch.argmax(outputs.start_logits, dim=1).cpu().tolist()
                        end_preds = torch.argmax(outputs.end_logits, dim=1).cpu().tolist()
                        start_labels = batch["start_positions"].cpu().tolist()
                        end_labels = batch["end_positions"].cpu().tolist()

                        all_start_preds.extend(start_preds)
                        all_end_preds.extend(end_preds)
                        all_start_labels.extend(start_labels)
                        all_end_labels.extend(end_labels)
                    except Exception as e:
                        batch_keys = list(batch.keys()) if isinstance(batch, dict) else "Unavailable"
                        print(f"[WARNING] Skipped batch {i} due to error: {e}. Batch keys: {batch_keys}")

            if len(all_start_preds) == 0:
                print(f"[WARNING] No valid predictions to evaluate at epoch {epoch}")
                return {
                    "val_loss": float("inf"),
                    "val_acc": 0.0
                }

            val_loss = total_loss / max(1, len(val_loader)) # 防止被0除

            start_acc = accuracy_score(all_start_labels, all_start_preds)
            end_acc = accuracy_score(all_end_labels, all_end_preds)
            avg_acc = (start_acc + end_acc) / 2.0

            self.writer.add_scalar("Accuracy/Start", start_acc, epoch)
            self.writer.add_scalar("Accuracy/End", end_acc, epoch)
            self.writer.add_scalar("Accuracy/Average", avg_acc, epoch)

            return {
                "val_loss": val_loss,
                "val_acc": avg_acc
            }

        except Exception as e:
            print(f"[ERROR] Evaluation failed at epoch {epoch}: {e}")
            return {
                "val_loss": float("inf"),
                "val_acc": 0.0
            }


        
    
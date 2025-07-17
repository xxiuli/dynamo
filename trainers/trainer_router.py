# trainers/trainer_router.py
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from trainers.base_trainer import BaseTrainer

class RouterTrainer(BaseTrainer):
    def __init__(self, model, config, device, tokenizer=None):
        super().__init__(model, config, device, tokenizer)

    def _forward_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']  # 即 task_id

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = F.cross_entropy(outputs, labels)
        return outputs, loss

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs, loss = self._forward_step(batch)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).detach().cpu().tolist()
                labels = batch["label"].detach().cpu().tolist()

                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_loss = total_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)

        return {
            "val_loss": avg_loss,
            "val_acc": acc
        }

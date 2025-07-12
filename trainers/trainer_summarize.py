# dataset_summarization.py
import torch
from trainers.base_trainer import BaseTrainer
from evaluate import load

class SummarizationTrainer(BaseTrainer):
    def __init__(self, model, config, device, tokenizer):
        super().__init__(model, config, device, tokenizer)

    def _forward_step(self, batch):
        outputs = self.model(**batch)
        return outputs, outputs.loss

    def evaluate(self, val_loader, epoch):
        debug_config = self.config.get("debug", {})
        skip_evaluation = debug_config.get("skip_evaluation", False)
        limit_batches = debug_config.get("limit_batches", None)

        if skip_evaluation:
            print("[DEBUG] Skipping evaluation step as per config.")
            return {
                "val_loss": 0.0,
                "val_rouge": 0.0
            }

        self.model.eval()
        total_loss = 0.0
        generated = []
        references = []

        # 每次调用都重新 load_metric，防止前后干扰
        self.metric = load("rouge")

        try:
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    # 如果本地调试，limit_batches=2， 那么就跑2个EPOACH就停了
                    if limit_batches is not None and i >= limit_batches:
                        break
                    try:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        total_loss += outputs.loss.item()

                        gen_tokens = self.model.generate(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            max_new_tokens=64
                        )

                        preds = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                        
                        labels = batch['labels'].clone()
                        labels[labels == -100] = self.tokenizer.pad_token_id
                        targets = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                        generated.extend(preds)
                        references.extend(targets)

                    except Exception as e:
                        print(f"[WARNING] Skipped batch {i} due to error: {e}")

            if len(generated) == 0 or len(references) == 0:
                print("[WARNING] No valid predictions or references to evaluate.")
                return {
                    "val_loss": float("inf"),
                    "val_rouge": 0.0
                }

            self.metric.add_batch(predictions=generated, references=references)
            rouge_scores = self.metric.compute()
            rouge_l = rouge_scores["rougeL"]

            self.writer.add_scalar("ROUGE/L", rouge_l, epoch)

            return {
                "val_loss": total_loss / max(1, len(val_loader)),
                "val_rouge": rouge_l
            }

        except Exception as e:
            print(f"[ERROR] Evaluation failed at epoch {epoch}: {e}")
            return {
                "val_loss": float("inf"),
                "val_rouge": 0.0
            }

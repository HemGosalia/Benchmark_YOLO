from ultralytics.models.yolo import YOLO
from ultralytics import YOLO
from ultralytics.engine.model import Model
import torch
import wandb
import os
# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize a new W&B run
wandb.init(project="yolo_buck_patched_benchmarks")

# Load the custom model configuration
model = YOLO('yolov9t.yaml')
model.model.to(device)

# Define a callback to log losses and additional metrics at the end of each training batch
def log_metrics(trainer):
    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
        loss_items = trainer.loss_items
        if len(loss_items) >= 3:
            wandb_log_data = {
                "train/box_loss": loss_items[0],
                "train/cls_loss": loss_items[1],
                "train/dfl_loss": loss_items[2],
                "epoch": getattr(trainer, "epoch", 0),
                "learning_rate": trainer.optimizer.param_groups[0]['lr'],
                "gpu_mem": torch.cuda.memory_reserved(device) / 1e9,  # More accurate GPU memory logging
                "GFLOPs": getattr(trainer.model, "flops", 0),  # Avoid attribute error
                "parameters": sum(p.numel() for p in trainer.model.parameters())
            }
            
            # Ensure trainer.metrics exists and is a dictionary before accessing its values
            if hasattr(trainer, "metrics") and isinstance(trainer.metrics, dict):
                wandb_log_data.update({
                    "metrics/precision": trainer.metrics.get("precision", 0),
                    "metrics/recall": trainer.metrics.get("recall", 0),
                    "metrics/mAP_50": trainer.metrics.get("mAP_50", 0),
                    "metrics/mAP_50-95": trainer.metrics.get("mAP_50-95", 0),
                    "val/box_loss": trainer.metrics.get("val/box_loss", 0),
                    "val/cls_loss": trainer.metrics.get("val/cls_loss", 0),
                    "val/dfl_loss": trainer.metrics.get("val/dfl_loss", 0),
                })

            wandb.log(wandb_log_data, step=wandb_log_data["epoch"])

        torch.cuda.empty_cache()

# Register the callback with the YOLO model
model.add_callback('on_train_batch_end', log_metrics)

# Train the model with the specified configuration and sync to W&B
result_final_model = model.train(
    data="/kaggle/input/bucktales-patched/dtc2023.yaml",
    epochs=70,
    batch=8,
    optimizer='auto',
    project='yolo_buck_patched_benchmarks',
    save=True,
    imgsz=1280,
    warmup_epochs=5,
    verbose=True
)

# Define model and dataset names
model_name = "yolov9t_vanilla"
dataset_name = "bucktales-patched"

# Save the model as .pth file in Kaggle workspace
save_path = f"/kaggle/working/models/{model_name}_{dataset_name}.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.model.state_dict(), save_path)
torch.cuda.empty_cache()

# Finish the W&B run
wandb.finish()

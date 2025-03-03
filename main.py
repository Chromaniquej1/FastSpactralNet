import torch
from sklearn.model_selection import train_test_split
from data_loader import load_pavia_university, preprocess_data
from dataset import PaviaUniversityDataset
from model import newFastViT
from trainer import calculate_latency_per_image, calculate_latency_per_batch, calculate_throughput, count_model_parameters
from metrics import overall_accuracy, average_accuracy, kappa_coefficient, calculate_f1_precision_recall, plot_confusion_matrix
from transformers import TrainingArguments, Trainer

# File paths for the dataset
image_file = "/kaggle/input/pavia-dataset2/PaviaU.mat"
gt_file = "/kaggle/input/pavia-dataset2/PaviaU_gt.mat"

# Load and preprocess data
image_data, ground_truth = load_pavia_university(image_file, gt_file)
spatial_spectral_data, y, label_encoder = preprocess_data(image_data, ground_truth)

# Split indices into train and test sets
train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)

# Create datasets
train_dataset = PaviaUniversityDataset(spatial_spectral_data[train_indices], y[train_indices])
test_dataset = PaviaUniversityDataset(spatial_spectral_data[test_indices], y[test_indices])

# Initialize the FastViT model
model = newFastViT(
    image_size=5,
    patch_size=1,
    num_channels=103,  # Pavia University has 103 spectral bands
    num_classes=len(np.unique(y)),
    embed_dim=768,
    depth=6,
    num_heads=12,
    mlp_ratio=4.
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# Create a data collator
data_collator = lambda data: {
    'x': torch.stack([d['x'] for d in data]),
    'labels': torch.stack([d['labels'] for d in data])
}

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()},
    data_collator=data_collator
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Generate predictions on the test dataset
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

# Calculate metrics
oa = overall_accuracy(y[test_indices], y_pred)
aa = average_accuracy(y[test_indices], y_pred)
kappa = kappa_coefficient(y[test_indices], y_pred)
f1, precision, recall = calculate_f1_precision_recall(y[test_indices], y_pred)
print(f"Overall Accuracy (OA): {oa:.4f}")
print(f"Average Accuracy (AA): {aa:.4f}")
print(f"Kappa Coefficient (κ): {kappa:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Calculate latency and throughput
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
latency_per_image = calculate_latency_per_image(model, test_loader, device)
latency_per_batch = calculate_latency_per_batch(model, test_loader, device)
throughput = calculate_throughput(model, test_loader, device)
print(f"Latency per image: {latency_per_image:.4f} ms")
print(f"Latency per batch: {latency_per_batch:.4f} seconds")
print(f"Throughput: {throughput:.2f} samples/second")

# Count the number of parameters
num_params = count_model_parameters(model)
print(f"Number of trainable parameters: {num_params:.2f} M")

# Plot confusion matrix
plot_confusion_matrix(y[test_indices], y_pred)
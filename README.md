# Fine-tune-LLM

<p align="center">
  <a href="https://colab.research.google.com/github/phuc123asd/Fine-tune-LLM/blob/main/finetune.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
  </a>
  <a href="https://github.com/phuc123asd/Fine-tune-LLM">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?logo=github" alt="GitHub Repo" />
  </a>
  <img src="https://img.shields.io/badge/QLoRA-4bit-blue" alt="QLoRA 4-bit" />
  <img src="https://img.shields.io/badge/PEFT-LoRA-green" alt="PEFT LoRA" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-orange" alt="Google Colab" />
</p>

A practical **LLM fine-tuning project** that shows how to train a conversational language model with **QLoRA 4-bit** on **Google Colab**, save the trained **LoRA adapter**, and reload it later for inference.

This repository is designed as a **hands-on starter project** for students and beginners who want to understand the real workflow of parameter-efficient fine-tuning without retraining a full model.

---

## Project overview

This project focuses on a lightweight fine-tuning pipeline using:

- **Google Colab** for accessible GPU training
- **BitsAndBytes 4-bit quantization** to reduce VRAM usage
- **PEFT / LoRA** to train only a small number of parameters
- **TRL SFTTrainer** for supervised fine-tuning
- **Google Drive** for storing the base model path and trained adapters

Instead of updating all model weights, the notebook attaches trainable LoRA modules to key transformer projection layers and saves the resulting adapter after training.

---

## Workflow

<p align="center">
  <img src="assets/workflow.png" alt="QLoRA workflow" width="900" />
</p>

---

## Repository structure

```bash
Fine-tune-LLM/
├── finetune.ipynb      # Main Colab notebook for QLoRA training and inference
└── README.md
```

---

## What the notebook does

The notebook walks through the complete pipeline:

1. Mount **Google Drive**
2. Create a storage directory for models and adapters
3. Install required libraries (`torch`, `transformers`, `accelerate`, `bitsandbytes`, `trl`, `peft`)
4. Upload a **JSONL dataset**
5. Load a base model in **4-bit NF4** mode
6. Prepare the model for **k-bit training**
7. Attach **LoRA adapters** to projection layers such as:
   - `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - `gate_proj`, `up_proj`, `down_proj`
8. Fine-tune with **SFTTrainer**
9. Save the trained adapter and tokenizer
10. Reload the adapter for inference / chat testing

---

## Tech stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- BitsAndBytes
- Google Colab
- Google Drive

---

## Training configuration

Current notebook configuration highlights:

```python
r = 8
lora_alpha = 16
lora_dropout = 0.05
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 2e-4
num_train_epochs = 3
fp16 = True
optim = "paged_adamw_8bit"
lr_scheduler_type = "cosine"
```

These settings make the project suitable for **limited-VRAM environments** such as common Colab GPUs.

---

## Dataset format

The notebook expects a dataset in **JSONL** format.

Example:

```json
{"text": "User: Hello\nAssistant: Hi! How can I help you today?"}
{"text": "User: Explain LoRA simply.\nAssistant: LoRA trains small adapter matrices instead of updating the entire model."}
```

Tips for better fine-tuning quality:

- Keep the style of the samples consistent
- Use clean and relevant conversation data
- Avoid noisy or contradictory responses
- Keep formatting uniform across all examples

---

## Open in Colab

Use the notebook directly in Colab:

**[Open finetune.ipynb in Google Colab](https://colab.research.google.com/github/phuc123asd/Fine-tune-LLM/blob/main/finetune.ipynb)**

---

## How to run

### 1. Open the notebook
Open `finetune.ipynb` in Google Colab.

### 2. Mount your Google Drive
This is used to:
- load the model from Drive
- save the fine-tuned LoRA adapter
- keep outputs after Colab runtime resets

### 3. Install dependencies
Run the setup cells to install the required packages.

### 4. Upload your dataset
Upload a `.jsonl` file and confirm the path used in the notebook.

### 5. Update your model path
Set the base model path, for example:

```python
model_path = "/content/drive/MyDrive/AI_models/models/snapshots/..."
```

### 6. Train
Run the training cells to start QLoRA fine-tuning.

### 7. Save the adapter
After training, the notebook saves the LoRA adapter and tokenizer to Drive.

### 8. Reload for inference
Load the adapter back into the base model and test the result with sample prompts.

---

## Expected outputs

After training, you should have:

- A trained **LoRA adapter**
- A saved **tokenizer**
- A reusable inference workflow
- A notebook-based demo suitable for portfolio presentation

---

## Why this project is good for a portfolio

This project is useful for showcasing:

- Understanding of **parameter-efficient fine-tuning**
- Practical use of **quantization** for low-resource training
- Experience with the **Hugging Face ecosystem**
- Ability to build a full **train → save → reload → test** pipeline
- Applied skills in **LLM engineering workflows** rather than theory alone

If you are building your CV, this repo can support bullets such as:

- Built a QLoRA-based LLM fine-tuning pipeline on Google Colab using PEFT, TRL, and BitsAndBytes
- Trained conversational adapters with 4-bit quantization to reduce VRAM requirements
- Implemented adapter saving and reloading workflow for lightweight deployment and testing

---

## Common issues

### 1. CUDA / VRAM errors
Try:
- lowering sequence length
- reducing batch size
- using a smaller base model
- keeping 4-bit quantization enabled

### 2. `trl` / `transformers` version mismatch
Use compatible versions and restart the runtime after installation.

### 3. Adapter loads but output looks like the base model
Make sure inference loads the **saved LoRA adapter**, not just the base model.

### 4. Dataset quality is poor
The model can only learn from the examples provided. Cleaner data usually gives better responses than adding more noisy data.

---

## Suggested future improvements

- Add example training logs or screenshots
- Add before/after sample outputs
- Add evaluation examples
- Add support for Hugging Face model download instead of only Drive-based paths
- Add a dedicated inference notebook or script
- Add adapter publishing instructions for Hugging Face Hub

---

## Acknowledgments

Built with open-source tools from:

- Hugging Face
- PEFT
- TRL
- BitsAndBytes
- PyTorch
- Google Colab

---

## Author

**Phuc Vo**

If you found this repository useful, consider starring it and using it as a base for your own LLM fine-tuning experiments.

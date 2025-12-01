------

**Experiment Overview**

 Considering the performance and computational cost of vision-language models (VLMs), we conducted our experiments based on the LLaVA-1.5-13B model to validate the effectiveness of the proposed strategy.

------

**Model Usage Instructions**

**Step 1:**
Design your own efficient **specialized vision module**. This module is highly pluggable and generalizable. In addition to the customized ResNet-18 and ConvNeXt-Base described in the paper, you can flexibly substitute more powerful vision encoders—such as those based on Transformer or Mamba architectures—to better capture both local and global information in EEG images, further enhancing the visual perception and processing capabilities of the VLM.

During training and testing, use the specialized vision module to extract **intermediate-layer features** (high-level representations) of shape `[N, 1, 1024]`. Save these features in `.npy` format in the `sleep_data` directory, saved as `train_high_level_features.npy` for training and `eval_high_level_features.npy` for testing. These files will be used for subsequent model fine-tuning and evaluation.

**Step 2:** 

Prepare your fine-tuning and evaluation data. Since the high-level representations (also known as feature maps) extracted by the specialized vision module are treated as images in the downstream process, you should modify the input images in the corresponding `json` or `jsonl` files to a list format: `[Image1, Image2]`. Here, `Image2` is identical to `Image1` and serves as a placeholder. During fine-tuning and evaluation, this placeholder will be automatically replaced by the corresponding `high_level_features` generated in Step 1.

This approach enables seamless integration with the existing VLM data processing pipeline, requiring only moderate modifications to the core code to achieve flexible feature injection and unified data handling.

**Step 3: Fine-tuning with LoRA**

```bash
deepspeed llava/train/train_mem.py \
  --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path your_checkpoints_path/LLaVA-v1.5-13b \
  --version v1 \
  --data_path sleep_data/eeg_ft.json \
  --image_folder sleep_data/eeg_images_ft \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ./checkpoints/your_checkpoints_save_path \
  --num_train_epochs 2 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 50000 \
  --save_total_limit 1 \
  --learning_rate 3e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 1 \
  --lazy_preprocess True \
  --report_to wandb
```

**Step 4: Evaluation**

```bash
python llava/eval/model_vqa.py \
  --model-path checkpoints/your_checkpoints_save_path/ \
  --model-base your_checkpoints_path/LLaVA-v1.5-13b/ \
  --question-file sleep_data/eeg_eval.jsonl \
  --image-folder sleep_data/eeg_images_eval \
  --answers-file sleep_data/answers/eeg_answers.jsonl 
```

------

**实验说明**
 考虑到视觉-语言模型（VLMs）的性能和计算成本，我们在 LLaVA-1.5-13B 基础上进行了实验，以验证所提出策略的有效性。

------

**模型使用说明**

**Step1：**
 设计你自己的高效 specialized vision module。本模块具有良好的即插即用性和通用性，除了论文提出的定制 ResNet-18 和 ConvNeXt-Base 外，也可灵活替换为如 Transformer、Mamba 等更强的视觉编码器，以更好地捕捉 EEG 图像中的局部与全局信息，进一步增强VLM的视觉感知及处理能力。

在训练及测试过程中，使用 specialized vision module 提取形状为 `[N, 1, 1024]` 的中间层特征（高层语义表示），并以 `.npy` 格式分别命名为 `train_high_level_features.npy`（训练集）和 `eval_high_level_features.npy`（测试集），保存在 `sleep_data` 文件夹中，用于后续的模型微调和评估。

**Step2：**

构建你的微调和评估数据。考虑到 specialized vision module 提取的 high-level representations（也称为特征图 feature maps），因此我们在下游流程中将其视为图像进行处理。具体来说，在对应的 `json` 或 `jsonl` 文件中，将输入图像修改为 `[Image1，Image2]` 的列表格式。其中 `Image2` 与 `Image1` 相同，作为占位图存在。在微调及评估过程中，该占位图会自动被 Step1 中生成的对应 `high_level_features` 替换。

这种做法可以无缝对接现有的 VLM 数据处理流程，仅需适量修改主干代码即可实现特征的灵活注入和统一的数据处理。

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


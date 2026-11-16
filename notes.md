# Notes

use the module `uv` to install the .venv and requirements .txt
install using 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # then restart your shell
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## current cli command used
```bash
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain/LEAD/P-11-b1024-p50-g16-learnable-3D/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-b1024-p50-g16-learnable-3D-F-ADFTD-Multi --model LEAD --data MultiDatasets \
--training_dataset OpenNeuro \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type 3D --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 4 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 20 --patience 15

```

## Loading Personal Dataset
Data set needs to be processed in proper way follow `data_preprocessing/ADFTD-PS/ADFTD-PS_preprocessing.ipynb` for example
processed dataset then needs to be placed in `dataset/` with folder of relevant name.

> ### Important
> You also need to create a Dataloader for the dataset
> main thing that needs to be implemented is how the data is split into training, validation, and testing
>
> Look at `data_provider/datset_loader/open_neuro_loader.py` for simple example
>
> Data loader then needs to be added to the `data_folder_dict` in `data_provider/data_loader.py`

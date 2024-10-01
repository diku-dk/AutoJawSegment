
## pretraining on OWIA dataset
mp init_project --name pretraining_IWOAI --data_dir ./datasets/pretrain_IWOAI
mp train --overwrite --num_GPUs=1 --force_GPU=0

## finetune
mp init_project --name fine_tune --data_dir ./datasets/rasmus

# first training own dataset using the model pretrained on OWIA dataset

mp train --overwrite --force_GPU=0 --epochs 100 --initialize_from "C:\\Users\\Administrator\\PycharmProjects\\AutoJawSegment\\pretraining_IWOAI\\model\\@epoch_170_val_dice_0.84136.h5"

# predictions
mp predict --num_GPUs=1 --force_GPU=0 --sum_fusion --overwrite --no_eval --by_radius --out_dir predictions

### continue training when having more data

mp train --continue --force_GPU=0 --epochs 80

### miscellaneous
mp predict --sum_fusion --overwrite --by_radius --no_eval --force_GPU=0 --test_dir "C:\Users\Administrator\PycharmProjects\AutoJawSegment\datasets\rasmus/train"
--out_dir predictions --use_diag_dim

mp predict --sum_fusion --overwrite --by_radius --no_eval  --out_dir predictions_no_arg --use_diag_dim --num_GPUs=0
mp predict --num_GPUs=1 --force_GPU=0 --sum_fusion --overwrite --no_eval --by_radius --out_dir predictions_2

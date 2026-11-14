# Flickr
python -m src.ml.train --dataset flickr --conv hgt \
    --hidden-channels 64 --num-heads 4 --num-layers 2 \
    --dropout 0.2 --conv-dropout 0.1 --edge-dropout 0.1 \
    --lr 1e-3 --weight-decay 1e-3 --wd-split \
    --batch-size 32 --epochs 30 --patience 6 \
    --scheduler cosine --grad-clip 1.0 --seed 42 --tag _flickr

# Digg
python -m src.ml.train --dataset digg --conv hgt \
    --hidden-channels 32 --num-heads 2 --num-layers 2 \
    --dropout 0.3 --input-dropout 0.1 --conv-dropout 0.3 --edge-dropout 0.2 \
    --lr 5e-4 --weight-decay 1e-2 --wd-split \
    --batch-size 32 --epochs 40 --patience 8 \
    --loss huber --scheduler plateau --grad-clip 1.0 --seed 42 --tag _digg

# Higgs
python -m src.ml.train --dataset higgs --conv hgt \
    --hidden-channels 64 --num-heads 4 --num-layers 2 \
    --dropout 0.2 --conv-dropout 0.1 --edge-dropout 0.15 \
    --lr 1e-3 --weight-decay 1e-3 --wd-split \
    --batch-size 64 --epochs 30 --patience 6 \
    --scheduler cosine --grad-clip 1.0 --seed 42 --tag _higgs


RGCN

python -m src.ml.train_rgcn --dataset_name flickr \
  --hidden_channels 128 --num_layers 2 --batch_size 64 \
  --lr 1e-3 --weight_decay 1e-4 --dropout 0.2 --edge_dropout 0.1 \
  --loss mse --epochs 60 --patience 15 --seed 42

python -m src.ml.train_rgcn --dataset_name digg \
  --hidden_channels 32 --num_layers 2 --batch_size 32 \
  --lr 5e-4 --weight_decay 1e-3 --dropout 0.5 --edge_dropout 0.2 \
  --loss huber --huber_delta 1.0 --epochs 100 --patience 20 --seed 42

python -m src.ml.train_rgcn --dataset_name higgs \
  --hidden_channels 64 --num_layers 2 --batch_size 64 \
  --lr 1e-3 --weight_decay 1e-4 --dropout 0.3 --edge_dropout 0.1 \
  --loss huber --huber_delta 1.0 --epochs 60 --patience 15 --seed 42


XGBOOST

python train_xgb.py --dataset_name flickr \
    --max_depth 6 --learning_rate 0.05 --n_estimators 3000 --early_stopping_rounds 100 \
    --min_child_weight 3 --subsample 0.8 --colsample_bytree 0.8 \
    --reg_alpha 0.1 --reg_lambda 5 --gamma 0.1


python train_xgb.py --dataset_name digg \
    --max_depth 3 --learning_rate 0.03 --n_estimators 3000 --early_stopping_rounds 100 \
    --min_child_weight 10 --subsample 0.7 --colsample_bytree 0.7 \
    --reg_alpha 1 --reg_lambda 10 --gamma 1

python train_xgb.py --dataset_name higgs \
    --max_depth 5 --learning_rate 0.05 --n_estimators 3000 --early_stopping_rounds 100 \
    --min_child_weight 5 --subsample 0.8 --colsample_bytree 0.8 \
    --reg_alpha 0.1 --reg_lambda 5 --gamma 0.5


python train_xgb.py --dataset_name flickr --tune_trials 40 --n_estimators 3000 --early_stopping_rounds 100
python train_xgb.py --dataset_name digg   --tune_trials 40 --n_estimators 3000 --early_stopping_rounds 100
python train_xgb.py --dataset_name higgs  --tune_trials 40 --n_estimators 3000 --early_stopping_rounds 100



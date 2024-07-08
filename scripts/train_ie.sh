export CUDA_VISIBLE_DEVICES=0

EXP_NO="test2"
MODALS="avl"

echo "IEMOCAP, ${MODALS}, ${EXP_NO}"

LOG_PATH="./logs/IEMOCAP/${MODALS}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u ./code/train.py \
--name ${EXP_NO} \
--modals ${MODALS} \
--dataset "IEMOCAP" \
--data_dir "./data/iemocap/IEMOCAP_features.pkl" \
--log_dir ${LOG_PATH}/${EXP_NO} \
--no_cuda \
--windowp 4 \
--windowf 4 \
--gnn_nhead 7 \
--lr 0.0003 \
--l2 0.0 \
--class_weight \
--gamma 0.05 \
--beta 0.0 \
--dropout 0.4 \
--tau 1 \
--epochs 60 \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1

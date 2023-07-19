# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory

# hard coded inputs
GPUID='1 2 3 4 6 7'
CONFIG=configs/attack.yaml
REPEAT=1
OVERWRITE=1

# OUTDIR=outputs-bce/${DATASET}/poison-10-task-target-2

###############################################################

# process inputs
# mkdir -p $OUTDIR

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!

# i=10
# NOISE_PATH=/home/ubuntu/Thesis/outputs-bce/cifar-100/draft/coda-p/triggers/repeat-1/task-trigger-gen/target-$i-noise_weight-100.npy
# OUTDIR=outputs-bce/${DATASET}/poison-10-task-target-$i-ns-100
# mkdir -p $OUTDIR
# python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.0 \
#     --noise_path $NOISE_PATH \
#     --target_lab 10 \
#     --log_dir ${OUTDIR}/coda-p

for i in 0 20 40 60 80
do
    NOISE_PATH=/home/ubuntu/Thesis/outputs-bce/cifar-100/draft/coda-p/triggers/repeat-1/task-trigger-gen/target-$i-noise_weight-100.npy
    OUTDIR=outputs-bce/${DATASET}/poison-10-task-target-$i-ns-100
    mkdir -p $OUTDIR
    python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name L2P \
        --prompt_param 30 20 -1 \
        --log_dir ${OUTDIR}/l2p++ \
        --noise_path $NOISE_PATH \
        --target_lab $i 
done

# for i in 60 70 80 90
# do
#     OUTDIR=outputs/${DATASET}/poison-10-task-target-$i
#     NOISE_PATH=/home/ubuntu/Thesis/outputs/cifar-100/attack/coda-p/triggers/repeat-1/task-trigger-gen/target-$i.npy
#     mkdir -p $OUTDIR
#     python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name L2P \
#         --prompt_param 30 20 -1 \
#         --log_dir ${OUTDIR}/l2p++ \
#         --target_lab $i \
#         --noise_path $NOISE_PATH 
# done

# i=10
# OUTDIR=outputs-bce/${DATASET}/test-10
# mkdir -p $OUTDIR
# NOISE_PATH=/home/ubuntu/Thesis/outputs-bce/cifar-100/draft/coda-p/triggers/repeat-1/task-trigger-gen/target-10-noise_weight-100.npy
# python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.0 \
#     --target_lab 10 \
#     --log_dir ${OUTDIR}/coda-p \
#     --noise_path $NOISE_PATH 
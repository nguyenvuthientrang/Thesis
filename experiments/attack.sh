# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory

# hard coded inputs
GPUID='4 7'
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

# NOISE_PATH=/home/ubuntu/Thesis/outputs-bce/cifar-100/draft/coda-p/triggers/repeat-1/task-trigger-gen/target-2-noise_weight-100-07-21-19_50_20.npy
# OUTDIR=outputs-bce/${DATASET}/poison-10-task-target-2-ns-100
# mkdir -p $OUTDIR
# python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.0 \
#     --noise_path $NOISE_PATH \
#     --log_dir ${OUTDIR}/coda-p



for i in 10 30 50 70 90
do
    OUTDIR=outputs/${DATASET}/poison-10-task-target-$i
    NOISE_PATH=/home/ubuntu/Thesis/outputs-bce/cifar-100/draft/coda-p/triggers/repeat-1/task-trigger-gen/target-$i-noise_weight-100.npy
    mkdir -p $OUTDIR
    python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name CODAPrompt \
        --prompt_param 100 8 0.0 \
        --log_dir ${OUTDIR}/coda-p \
        --target_lab $i \
        --noise_path $NOISE_PATH 
done

i=10
# OUTDIR=outputs-bce/${DATASET}/test-10
# mkdir -p $OUTDIR
# NOISE_PATH=/home/ubuntu/Thesis/outputs-bce/cifar-100/draft/coda-p/triggers/repeat-1/task-trigger-gen/target-10-noise_weight-100.npy
# python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.0 \
#     --target_lab 10 \
#     --log_dir ${OUTDIR}/coda-p \
#     --noise_path $NOISE_PATH 
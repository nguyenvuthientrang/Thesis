# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory

# hard coded inputs
GPUID='5 6 7'
CONFIG=configs/attack.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
# mkdir -p $OUTDIR

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!

# python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.0 \
#     --log_dir ${OUTDIR}/coda-p

for i in 60 70 80 90
do
    OUTDIR=outputs/${DATASET}/poison-10-task-target-$i
    NOISE_PATH=/home/ubuntu/Thesis/outputs/cifar-100/attack/coda-p/triggers/repeat-1/task-trigger-gen/target-$i.npy
    mkdir -p $OUTDIR
    python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name L2P \
        --prompt_param 30 20 -1 \
        --log_dir ${OUTDIR}/l2p++ \
        --target_lab $i \
        --noise_path $NOISE_PATH 
done


# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=100

# save directory

# hard coded inputs
GPUID='2 3 7'
CONFIG=configs/attack.yaml
REPEAT=1
OVERWRITE=1

###############################################################

# process inputs
# mkdir -p $OUTDIR

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!

# for i in 0 40 50 90
# do
#     OUTDIR=outputs-imgnet/${DATASET}/attack-$i
#     mkdir -p $OUTDIR
#     python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name CODAPrompt \
#         --prompt_param 100 8 0.0 \
#         --finetune \
#         --target_lab $i \
#         --log_dir ${OUTDIR}/coda-p
# done

# for i in 0 40 50 90
# do
#     OUTDIR=outputs-imgnet/${DATASET}/poison-10-task-target-$i
#     NOISE_PATH=/home/ubuntu/Thesis/outputs-imgnet/cifar-100/attack-$i/coda-p/triggers/repeat-1/task-trigger-gen/target-$i.npy
#     mkdir -p $OUTDIR
#     python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name CODAPrompt \
#         --prompt_param 100 8 0.0 \
#         --log_dir ${OUTDIR}/coda-p \
#         --target_lab $i \
#         --noise_path $NOISE_PATH 
# done


OUTDIR=outputs-imgnet/${DATASET}/poison-1-task-target-10
NOISE_PATH=/home/ubuntu/Thesis/outputs-imgnet/cifar-100/attack-0/coda-p/triggers/repeat-1/task-trigger-gen/target-0-07-13-02_19_23.npy
mkdir -p $OUTDIR
python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ${OUTDIR}/coda-p \
    --target_lab 10 \
    --noise_path $NOISE_PATH 

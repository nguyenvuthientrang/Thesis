# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory

# hard coded inputs
GPUID='1 7'
CONFIG=configs/attack.yaml
REPEAT=1
OVERWRITE=1

###############################################################

i=10
    OUTDIR=outputs/${DATASET}/poison-10-task-target-$i
    NOISE_PATH=/home/ubuntu/Thesis/outputs-bce/cifar-100/draft/coda-p/triggers/repeat-1/task-trigger-gen/target-$i-noise_weight-100.npy
    mkdir -p $OUTDIR
    python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name CODAPrompt \
        --prompt_param 100 8 0.0 \
        --log_dir ${OUTDIR}/coda-p \
        --target_lab $i \
        --noise_path $NOISE_PATH 
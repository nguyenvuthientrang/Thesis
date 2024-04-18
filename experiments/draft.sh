# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory

# hard coded inputs
GPUID='1 2 6 7'
CONFIG=configs/attack.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# # process inputs
OUTDIR=outputs/${DATASET}/trigger-gen
mkdir -p $OUTDIR

# # CODA-P
# #
# # prompt parameter args:
# #    arg 1 = prompt component pool size
# #    arg 2 = prompt length
# #    arg 3 = ortho penalty loss weight - with updated code, now can be 0!

python3 -u draft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ${OUTDIR}/coda-p \
    --surrogate_dir outputs/cifar-100/warmup/coda-p/models/repeat-1/task-warmup/


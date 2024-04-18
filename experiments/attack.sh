# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory

# hard coded inputs
GPUID='1 2'
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

python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ${OUTDIR}/coda-p \
    --surrogate_dir outputs/cifar-100/warmup/coda-p/models/repeat-1/task-warmup/


# python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 \
#     --log_dir ${OUTDIR}/l2p++ \
#     --surrogate_dir outputs/cifar-100/warmup/coda-p/models/repeat-1/task-warmup/

# for i in 2 42 52 92
# do
#     OUTDIR=outputs_coda/${DATASET}/poison-10-task-target-$i
#     mkdir -p $OUTDIR

#     python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name CODAPrompt \
#         --prompt_param 100 8 0.0 \
#         --log_dir ${OUTDIR}/coda-p \
#         --surrogate_dir outputs/cifar-100/task-surrogate/coda-p/models/repeat-1/task-surrogate/ \
#         --target_lab $i

#     # python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     #     --learner_type prompt --learner_name L2P \
#     #     --prompt_param 30 20 -1 \
#     #     --log_dir ${OUTDIR}/l2p++ \
#     #     --surrogate_dir outputs/cifar-100/task-surrogate/coda-p/models/repeat-1/task-surrogate/ \
#     #     --target_lab $i 

# done

# for i in 2 42 52 92
# do
#     OUTDIR=outputs_l2p/${DATASET}/poison-10-task-target-$i
#     mkdir -p $OUTDIR

#     python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name CODAPrompt \
#         --prompt_param 100 8 0.0 \
#         --log_dir ${OUTDIR}/coda-p \
#         --surrogate_dir outputs/cifar-100/task-surrogate/l2p++/models/repeat-1/task-surrogate/ \
#         --target_lab $i

#     python3 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name L2P \
#         --prompt_param 30 20 -1 \
#         --log_dir ${OUTDIR}/l2p++ \
#         --surrogate_dir outputs/cifar-100/task-surrogate/l2p++/models/repeat-1/task-surrogate/ \
#         --target_lab $i 

# done

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


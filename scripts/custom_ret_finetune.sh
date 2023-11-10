# Choose DATA_TYPE from ["multisentence", "onesentence"]
DATA_TYPE="multisentence" \
WANDB_MODE="offline" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run \
--master_port 2502 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${ANNOTATION_PATH} \
--video_path ${VIDEOPATH} \
--datatype $DATA_TYPE \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--stage generation \
--diffusion_steps 50 \
--noise_schedule cosine \
--init_model output/custom_pretrain/best.pth \
--output_dir output/custom_finetune
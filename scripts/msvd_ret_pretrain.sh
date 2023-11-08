CUDA_VISIBLE_DEVICES=0 \
taskset -c 17-35 \
python -m torch.distributed.run \
--master_port 2502 \
--nproc_per_node=1 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 10 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path /data/kylee/projects/HBI/MSVD/msvd_data \
--video_path /data/kylee/projects/HBI/MSVD/msvd_data/MSVD_Videos \
--datatype msvd \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--stage discrimination \
--output_dir output/msvd_pretrain
# Tinyllama
# max_length=2048 
python3 -u train_wrapper.py > rst.txt
python3 -u train_wrapper.py --finetune_type qlora > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --hook --activation > rst.txt
python3 -u train_multithread.py --finetune_type qlora --autocast_context --hook --activation --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight > rst.txt
python3 -u train_wrapper.py --hook --activation > rst.txt
python3 -u train_multithread.py --hook --activation > rst.txt
python3 -u train_multithread.py --hook --weight > rst.txt
python3 -u train_multithread.py --hook --activation --asynchronous > rst.txt
python3 -u train_multithread.py --hook --weight --activation --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --activation > rst.txt

# max_length=512
python3 -u train_wrapper.py --max_length 512 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --max_length 512 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --max_length 512 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --max_length 512 --hook --activation > rst.txt
python3 -u train_multithread.py --finetune_type qlora --autocast_context --max_length 512 --hook --activation --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --max_length 512 > rst.txt
python3 -u train_wrapper.py --hook --activation --max_length 512 > rst.txt
python3 -u train_multithread.py --hook --activation --max_length 512 --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --max_length 512 > rst.txt

# max_length=2048, bs=3
python3 -u train_wrapper.py --batch_size 3 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --batch_size 3 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --batch_size 3 > rst.txt
python3 -u train_multithread.py --finetune_type qlora --autocast_context --batch_size 3 --hook --activation --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --batch_size 3 > rst.txt
python3 -u train_wrapper.py --hook --activation --batch_size 3 > rst.txt
python3 -u train_multithread.py --hook --activation --asynchronous --batch_size 3 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --batch_size 3 > rst.txt
python3 -u train_multithread.py --hook --weight --activation --asynchronous --batch_size 3 > rst.txt


# max_length=2048, bs=5
# python3 -u train_wrapper.py --batch_size 5 > rst.txt
# python3 -u train_wrapper.py --finetune_type qlora --batch_size 5 > rst.txt
# python3 -u train_wrapper.py --finetune_type qlora --autocast_context --batch_size 5 > rst.txt
# python3 -u train_wrapper.py --hook --weight --batch_size 5 > rst.txt
# python3 -u train_wrapper.py --hook --activation --batch_size 5 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --batch_size 5 > rst.txt



# Mistral
# max_length=512 
python3 -u train_wrapper.py --model /opt/models/hf/Mistral-7B-v0.1 --max_length 512 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/Mistral-7B-v0.1 --max_length 512 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Mistral-7B-v0.1 --max_length 512 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Mistral-7B-v0.1 --max_length 512 --hook --activation > rst.txt
python3 -u train_multithread.py --finetune_type qlora --autocast_context --model /opt/models/hf/Mistral-7B-v0.1 --max_length 512 --hook --activation --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 512 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 512 > rst.txt
python3 -u train_multithread.py --hook --activation --model /opt/models/hf/Mistral-7B-v0.1 --max_length 512 --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 512 > rst.txt

# max_length=1024 
python3 -u train_wrapper.py --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1024 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1024 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1024 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1024 --hook --activation > rst.txt
python3 -u train_multithread.py --finetune_type qlora --autocast_context  --hook --activation --model /opt/models/hf/Mistral-7B-v0.1  --asynchronous --max_length 1024 > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 1024 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1024 > rst.txt
python3 -u train_multithread.py --hook --activation --model /opt/models/hf/Mistral-7B-v0.1  --asynchronous --max_length 1024 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 1024 > rst.txt

# max_length=1600 
python3 -u train_wrapper.py --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1600 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1600 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1600 > rst.txt
python3 -u train_multithread.py --finetune_type qlora --autocast_context --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1600 --hook --activation --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 1600 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Mistral-7B-v0.1 --max_length 1600 > rst.txt
python3 -u train_multithread.py --hook --activation --model /opt/models/hf/Mistral-7B-v0.1  --asynchronous --max_length 1600 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 1600 > rst.txt

# max_length=2048 
python3 -u train_wrapper.py --model /opt/models/hf/Mistral-7B-v0.1 --max_length 2048 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/Mistral-7B-v0.1 --max_length 2048 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Mistral-7B-v0.1 --max_length 2048 > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 2048 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 2048 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Mistral-7B-v0.1 --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Mistral-7B-zstd-compressed-weights/level21 --max_length 2048 > rst.txt



# Gemma-2-2b-it
# 2048
python3 -u train_wrapper.py --model /opt/models/hf/gemma-2-2b-it > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/gemma-2-2b-it > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/gemma-2-2b-it --round 1 > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/gemma-2-2b-it --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/gemma-2-2b-it-zstd-compressed-weights/level21 --round 1  > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/gemma-2-2b-it --round 1  > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/gemma-2-2b-it --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/gemma-2-2b-it-zstd-compressed-weights/level21 --round 1  > rst.txt

# 1024
python3 -u train_wrapper.py --model /opt/models/hf/gemma-2-2b-it --max_length 1024 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/gemma-2-2b-it --max_length 1024 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/gemma-2-2b-it --max_length 1024 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/gemma-2-2b-it --max_length 1024 --hook --activation > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/gemma-2-2b-it --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/gemma-2-2b-it-zstd-compressed-weights/level21 --max_length 1024 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/gemma-2-2b-it --max_length 1024 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/gemma-2-2b-it --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/gemma-2-2b-it-zstd-compressed-weights/level21 --max_length 1024 > rst.txt



# Llama-3.1-8B，注意，使用老版本的 inject_from_files
# 512
python3 -u train_wrapper.py --model /opt/models/Llama-3.1-8B --max_length 512 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/Llama-3.1-8B --max_length 512 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/Llama-3.1-8B --max_length 512 > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/Llama-3.1-8B --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.1-8B-zstd-compressed-weights/level21 --max_length 512 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/Llama-3.1-8B --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.1-8B-zstd-compressed-weights/level21 --max_length 512 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/Llama-3.1-8B --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.1-8B-zstd-compressed-weights/level21 --max_length 512 > rst.txt


# 1024
python3 -u train_wrapper.py --model /opt/models/Llama-3.1-8B --max_length 1024 --round 1 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/Llama-3.1-8B --max_length 1024 --round 1 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/Llama-3.1-8B --max_length 1024 > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/Llama-3.1-8B --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.1-8B-zstd-compressed-weights/level21 --max_length 1024 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/Llama-3.1-8B --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.1-8B-zstd-compressed-weights/level21 --max_length 1024 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/Llama-3.1-8B --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.1-8B-zstd-compressed-weights/level21 --max_length 1024 > rst.txt



# Llama-3.2-3B-Instruct
# 1024
python3 -u train_wrapper.py --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 1024 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 1024 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 1024 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 1024 --hook --activation > rst.txt
python3 -u train_multithread.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 1024 --hook --activation --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 1024 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 1024 > rst.txt
python3 -u train_multithread.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 1024 --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 1024 > rst.txt

# 2048
python3 -u train_wrapper.py --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 2048 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 2048 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 2048 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 2048 --hook --activation > rst.txt
python3 -u train_multithread.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 2048 --hook --activation --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 2048 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 2048 > rst.txt
python3 -u train_multithread.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 2048 --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 2048 --layerwise > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 2048 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 2048 --layerwise > rst.txt

# 3000
python3 -u train_wrapper.py --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3000 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3000 --round 1 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3000 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3000 --hook --activation --round 1 > rst.txt
python3 -u train_multithread.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3000 --hook --activation --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 3000 --round 1 > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3000 --round 1 > rst.txt
python3 -u train_multithread.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3000 --asynchronous > rst.txt
python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3000 --layerwise --round 1 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 3000 --round 1 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 3000 --layerwise --round 1 > rst.txt

# 3500
# python3 -u train_wrapper.py --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3500 --round 1 > rst.txt
# python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3500 --round 1 > rst.txt
# python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3500 --round 1 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3500 --hook --activation --round 1 > rst.txt
# python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 3500 --round 1 > rst.txt
# python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 3500 --round 1 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 3500 --round 1 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 3500 --layerwise --round 1 > rst.txt

# 4096
# python3 -u train_wrapper.py --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 4096 --round 1 > rst.txt
# python3 -u train_wrapper.py --finetune_type qlora --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 4096 --round 1 > rst.txt
# python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 4096 --round 1 > rst.txt
python3 -u train_wrapper.py --finetune_type qlora --autocast_context --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 4096 --hook --activation --round 1 > rst.txt
# python3 -u train_wrapper.py --hook --weight --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 4096 --round 1 > rst.txt
# python3 -u train_wrapper.py --hook --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --max_length 4096 --round 1 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 4096 --round 1 > rst.txt
python3 -u train_wrapper.py --hook --weight --activation --model /opt/models/hf/Llama-3.2-3B-Instruct --outdir /opt/finetune/MemRift/weight_comp/prepare_weight/Llama-3.2-3B-Instruct-zstd-compressed-weights/level21 --max_length 4096 --layerwise --round 1 > rst.txt

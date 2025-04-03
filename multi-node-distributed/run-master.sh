NCCL_SOCKET_IFNAME=enp4s0 torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr=192.168.0.8 \
    --master_port=1235 \
    main.py \
        --run_type fsdp --num_loops 10
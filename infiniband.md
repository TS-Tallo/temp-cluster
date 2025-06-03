# VLLM
### **On Node 1 (Master, node-rank=0):**
``` bash
ray start --head --port=6379 --num-cpus=32 --num-gpus=2

docker run --gpus all --network host --rm \
  -e NCCL_SOCKET_IFNAME=enp129s0f0np0,enp129s0f1np1 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1 \
  -e NCCL_IB_TIMEOUT=23 \
  -e NCCL_IB_RETRY_CNT=7 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_NET_GDR_LEVEL=5 \
  -v /models:/data \
  vllm/vllm-openai:v0.8.5 \
  --model google/gemma-7b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --deployment-mode ray \
  --ray-address="auto"
```
### **On Node 2 (Worker, node-rank=1):**
``` bash
ray start --address='192.168.100.1:6379' --num-cpus=32 --num-gpus=2
```

# HF-TGI
### **Master Node (Node 1)**
``` bash
docker run --gpus all --network host --rm \
  -e NCCL_SOCKET_IFNAME=enp129s0f0np0,enp129s0f1np1 \
  -v /models:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id google/gemma-7b \
  --num-shard 2 \
  --sharded true \
  --tensor-parallel 2 \
  --tp-gpus-per-node 1 \
  --hostname 192.168.100.1 \
  --distributed-init-method "tcp://192.168.100.1:9000"
```
### **Worker Node (Node 2, for example if its IP is `192.168.100.2`)**
``` bash
docker run --gpus all --network host --rm \
  -e NCCL_SOCKET_IFNAME=enp129s0f0np0,enp129s0f1np1 \
  -v /models:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id google/gemma-7b \
  --num-shard 2 \
  --sharded true \
  --tensor-parallel 2 \
  --tp-gpus-per-node 1 \
  --hostname 192.168.100.2 \
  --distributed-init-method "tcp://192.168.100.1:9000"
```





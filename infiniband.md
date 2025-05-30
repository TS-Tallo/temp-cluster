# VLLM
### **On Node 1 (Master, node-rank=0):**
``` bash
docker run --gpus all --network host --rm \
  -e NCCL_SOCKET_IFNAME=enp129s0f0np0,enp129s0f1np1 \
  -v /models:/data \
  vllm/vllm-openai:v0.3.3 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed \
    --tensor-parallel-size 2 \
    --node-rank 0 \
    --master-addr 192.168.100.1 \
    --master-port 9000
```
### **On Node 2 (Worker, node-rank=1):**
``` bash
docker run --gpus all --network host --rm \
  -e NCCL_SOCKET_IFNAME=enp129s0f0np0,enp129s0f1np1 \
  -v /models:/data \
  vllm/vllm-openai:v0.3.3 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed \
    --tensor-parallel-size 2 \
    --node-rank 1 \
    --master-addr 192.168.100.1 \
    --master-port 9000
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





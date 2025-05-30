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

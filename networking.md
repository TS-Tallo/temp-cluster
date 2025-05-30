## 1. **Simple: Assign Separate IPs (Recommended for Simplicity/Testing)**
Assign a different IP address to each interface on both servers.
### **On Server 1**
``` bash
sudo ip link set enp129s0f0np0 up
sudo ip link set enp129s0f1np1 up
sudo ip addr add 192.168.100.1/24 dev enp129s0f0np0
sudo ip addr add 192.168.101.1/24 dev enp129s0f1np1
```
### **On Server 2**
``` bash
sudo ip link set enp129s0f0np0 up
sudo ip link set enp129s0f1np1 up
sudo ip addr add 192.168.100.2/24 dev enp129s0f0np0
sudo ip addr add 192.168.101.2/24 dev enp129s0f1np1
```
You now have two “parallel” connections:
- enp129s0f0np0 <-> enp129s0f0np0: 192.168.100.1 <-> 192.168.100.2
- enp129s0f1np1 <-> enp129s0f1np1: 192.168.101.1 <-> 192.168.101.2

#### **To Use in NCCL/Deep Learning:**
Set both interface names (comma-separated) so NCCL can use both:
``` bash
export NCCL_SOCKET_IFNAME="enp129s0f0np0,enp129s0f1np1"
```
## 2. **Advanced: Bond (Aggregate) the Links**
### **A. Netplan YAML for Bonded Interfaces**
Edit `/etc/netplan/01-bond.yaml` (example):
``` yaml
network:
  version: 2
  bonds:
    bond0:
      interfaces: [enp129s0f0np0, enp129s0f1np1]
      addresses:
        - 192.168.200.1/24        # Use .2 for server 2
      mtu: 9000
      parameters:
        mode: balance-rr          # Or 802.3ad for LACP with supported switch
        mii-monitor-interval: 100
  ethernets:
    enp129s0f0np0: {}
    enp129s0f1np1: {}
```
On **server 2** use `192.168.200.2/24`.
### **B. Apply Netplan Configuration**
``` bash
sudo netplan apply
```
### **C. Test Bonding**
``` bash
ip addr show bond0
ping 192.168.200.2     # Between both servers
```
### **D. Use the Bond with NCCL/Apps**
``` bash
export NCCL_SOCKET_IFNAME=bond0
```
## **Tips**
- **Jumbo frames:** add `mtu: 9000` for maximal throughput
- **Choose bond mode:** `balance-rr` is easiest for direct attach; `802.3ad` requires switch support (not needed for point-to-point)
- **You can use separate IPs or bonding—prefer bonding for true bandwidth aggregation or redundancy**

## **Summary Table**

|  | IP on Server 1 | IP on Server 2 | Use in NCCL |
| --- | --- | --- | --- |
| Port 0 | 192.168.100.1 | 192.168.100.2 | Both singles, or bond0 |
| Port 1 | 192.168.101.1 | 192.168.101.2 |  |
| **Bonded** | 192.168.200.1 (bond0) | 192.168.200.2 (bond0) | bond0 |

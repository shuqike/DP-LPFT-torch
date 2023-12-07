cd models
mkdir -p checkpoint
cd checkpoint
wget -nc https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar
wget -nc https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
wget -nc https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/r-50-100ep.pth.tar
wget -nc https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/r-50-300ep.pth.tar
wget -nc https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar

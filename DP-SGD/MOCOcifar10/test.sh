# 1. debug state_dict loading
# python main.py --name test --epoch 4 --lp-epoch 2 --weights models/checkpoint/mocov2_rn50_800ep.pth.tar --debug
# python main.py --name test --epoch 4 --lp-epoch 2 --debug
# 2. debug linear probing freezing
python main.py --name test --epoch 4 --lp-epoch 2 --debug
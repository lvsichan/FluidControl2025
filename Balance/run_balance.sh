# without EDG
CUDA_VISIBLE_DEVICES=0 python bellman_main.py --policy 'bellman_SD3' --env 'solid_balance_2b_v'  --steps 2e6 --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --seed 10 --beta 10

# with EDG
CUDA_VISIBLE_DEVICES=0 python BSD3_EDG_main_2b.py --policy 'BSD3_EDG' --env 'solid_balance_2b_v'  --steps 2e6 --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --seed 0 --beta 10 --guide_p 0.3
# run Scoop balls 
# TD3
CUDA_VISIBLE_DEVICES=0 python main.py --policy 'TD3' --env 'scoop_balls'  --steps 5e6 --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --seed 8011 --beta 10
# SD3
CUDA_VISIBLE_DEVICES=0 python main.py --policy 'SD3' --env 'scoop_balls'  --steps 5e6 --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --seed 8011 --beta 10
# Ours
CUDA_VISIBLE_DEVICES=0 python bellman_main.py --policy 'bellman_SD3' --env 'scoop_balls'  --steps 5e6 --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --seed 8011 --beta 10



# run Scoop good balls with HER
#TD3
CUDA_VISIBLE_DEVICES=0 python main_her.py --policy 'SD3_her' --env 'scoop_good_balls'  --steps 5e6 --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --seed 8011 --beta 10
#SD3
CUDA_VISIBLE_DEVICES=0 python main_her.py --policy 'TD3_her' --env 'scoop_good_balls'  --steps 5e6 --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --seed 8011 --beta 10
#Ours
CUDA_VISIBLE_DEVICES=0 python main_her_bellman.py --policy 'bellman_SD3_her' --env 'scoop_good_balls'  --steps 5e6 --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --seed 8011 --beta 10
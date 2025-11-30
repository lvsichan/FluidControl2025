import numpy as np
import numpy as np
import torch
import gym
import argparse
import os
import utils
import random
from tensorboardX import SummaryWriter 

import TD3
import SD3
import bellman_SD3

import her_v2

import datetime
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="SD3")
    parser.add_argument("--env", default="scoop_ball")#  solid_balance   
    parser.add_argument("--noise_type", default="pink")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=80e3, type=int, help='Number of steps for the warm-up stage using random policy')
    parser.add_argument("--eval_freq", default=5e3, type=int, help='Number of steps per evaluation')
    parser.add_argument("--steps", default=1e6, type=float, help='Maximum number of steps')
    parser.add_argument("--max_size", default=5e6, type=float, help='Replay buffer size')

    parser.add_argument("--discount", default=0.99, help='Discount factor')
    parser.add_argument("--tau", default=0.005, help='Target network update rate')                    
    
    parser.add_argument("--actor_lr", default=1e-3, type=float)     
    parser.add_argument("--critic_lr", default=1e-3, type=float)    
    parser.add_argument("--hidden_sizes", default='400,300', type=str)  
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic

    parser.add_argument("--save_model", default=True)        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise

    parser.add_argument("--policy_freq", default=2, type=int, help='Frequency of delayed policy updates')

    parser.add_argument('--beta', default='best', help='The parameter beta in softmax')
    parser.add_argument('--num-noise-samples', type=int, default=50, help='The number of noises to sample for each next_action')
    parser.add_argument('--imps', type=int, default=0, help='Whether to use importance sampling for gaussian noise when calculating softmax values')
    # DRAC
    parser.add_argument("--qweight", default=0.2, type=float, help='The weighting coefficient that correlates value estimation from double actors')
    parser.add_argument("--reg", default=0.005, type=float, help='The regularization parameter for DARC')     
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}_{args.beta}_{args.noise_type}"
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("---------------------------------------")
    
    # SummaryWriter encapsulates everything 
    now = datetime.datetime.now()
    writer = SummaryWriter('runs/'+ f"{args.env}_{args.policy}_{args.seed}_{args.beta}_{args.noise_type}_"+now.strftime("%Y-%m-%d_%H-%M-%S")) 

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if args.env =="scoop_good_balls":
        from scoop_good_balls.Scoop_the_jelly import ScoopJellyEnvs
        from scoop_good_balls.AE_CNN import AutoEncoder
        import taichi as ti
        import numpy as np
        env = ScoopJellyEnvs(render=False)

    # env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_sizes": [int(hs) for hs in args.hidden_sizes.split(',')],
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "device": device,
    }

    if args.policy == "TD3_her":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq

        policy = TD3.TD3(**kwargs)
    elif args.policy == "SD3_her":
        env_beta_map = {
            'Ant-v2': 0.001,
            'BipedalWalker': 0.05,
            'HalfCheetah': 0.005,
            'Hoppper': 0.05,
            'LunarLanderContinuous': 0.5,
            'Walker2d': 0.1,
            'Humanoid': 0.05,
            'Swimmer': 500.0,
            'solid_balance':0.05,
            'scoop_ball':0.05,
        }

        kwargs['beta'] = env_beta_map[args.env] if args.beta == 'best' else float(args.beta)
        kwargs['with_importance_sampling'] = args.imps
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs['num_noise_samples'] = args.num_noise_samples

        policy = SD3.SD3(**kwargs)
   

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer_her = her_v2.ReplayBuffer(state_dim, action_dim, device,max_size=int(args.max_size))

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    #pink_noise
    from pink import PinkActionNoise
    seq_len = env._max_episode_steps
    action_dim = env.action_space.shape[-1]
    # print(max_action)
    noise_scale = max_action*args.expl_noise
    noise = PinkActionNoise(noise_scale, seq_len, action_dim)

    best_score = -np.inf
    flag = False
    her_flag = False
    her_timesteps = 0
    state_list, action_list, next_state_list, reward_list, done_bool_list,isIn_list = [],[],[],[],[],[]
    for t in range(int(args.steps)):
        episode_timesteps += 1

        # select action randomly or according to policy
        if t < args.start_timesteps:
            action = (max_action - min_action) * np.random.random(env.action_space.shape) + min_action
        else:
            action = (
                policy.select_action(np.array(state)) 
                # + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                + noise()
            ).clip(-max_action, max_action)

        next_state, reward, done, isIn = env.step(action) 
        # print(next_state[4])
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        replay_buffer_her.add(state, action, next_state, reward, done_bool)


        state_list.append(state)
        action_list.append(action)
        next_state_list.append(next_state)
        reward_list.append(reward)
        done_bool_list.append(done_bool)
        isIn_list.append(isIn)

        state = next_state
        episode_reward += reward
        # print("第-1次")
        if t >= args.start_timesteps:
            # print("第"+str(t)+"次")
            policy.train(replay_buffer_her, args.batch_size)
        

        if (t + 1) % args.eval_freq == 0:
            flag = True

        if env.flag:
            her_flag = True
        if done: 
            # her signal
            if her_flag:
                replay_buffer_her.store_eposide(state_list, action_list, next_state_list, reward_list, done_bool_list,isIn_list)
                print("Complete HER!")
            state_list, action_list, next_state_list, reward_list, done_bool_list,isIn_list = [],[],[],[],[],[]

            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            writer.add_scalar('episode_reward', episode_reward, t, walltime=time.time()) 
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            writer.add_scalar('keep_time', episode_timesteps, t, walltime=time.time()) 

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            her_flag = False
            # Evaluate episode
            if flag:
                eval_episodes=10
                avg_reward = 0.
                for _ in range(eval_episodes):
                    state, done = env.reset(), False
                    while not done:
                        action = policy.select_action(np.array(state))
                        state, reward, done, _ = env.step(action)
                        avg_reward += reward
                state, done = env.reset(), False
                avg_reward /= eval_episodes
                writer.add_scalar('eval_reward', avg_reward, t, walltime=time.time()) 
                print("---------------------------------------")
                print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")
                if args.save_model and avg_reward>best_score: 
                    best_score = avg_reward
                    policy.save(f"./models/{file_name}")
                    print("best model saved!!!")
                flag = False
                



import numpy as np
import numpy as np
import torch
import gym
import argparse
import os
import BSD3_EDG_utils
import random
from tensorboardX import SummaryWriter 

import BSD3_EDG

import datetime
import time

from guided_utils import compute_domain_reward_2b

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="SD3")
    parser.add_argument("--env", default="scoop_ball")#  solid_balance   
    parser.add_argument("--noise_type", default="pink")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=80e3, type=int, help='Number of steps for the warm-up stage using random policy')
    parser.add_argument("--eval_freq", default=25e3, type=int, help='Number of steps per evaluation')
    parser.add_argument("--steps", default=1e6, type=float, help='Maximum number of steps')
    parser.add_argument("--max_size", default=1e6, type=float, help='Replay buffer size')

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
    
    #EDFG
    parser.add_argument("--guide_p", default=0.3, type=float)              # Noise added to target policy during critic update

    
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}_{args.beta}_{args.noise_type}_{args.guide_p}"
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("---------------------------------------")

    # SummaryWriter encapsulates everything 
    now = datetime.datetime.now()
    writer = SummaryWriter('runs_asia_v2/'+ f"{args.env}_{args.policy}_{args.seed}_{args.beta}_{args.noise_type}_{args.guide_p}_"+now.strftime("%Y-%m-%d_%H-%M-%S")) 

    if args.save_model and not os.path.exists("./models_asia_v2"):
        os.makedirs("./models_asia_v2")

    if  args.env =="solid_balance_2b_v":
        from solid_balance_v6.solid_balance_v import MusicPipeEnvs
        from solid_balance_v6.AE_CNN import AutoEncoder
        import taichi as ti

        n_ball = 2
        n_pipe = 1

        env = MusicPipeEnvs(n_ball,n_pipe,False)
        env._max_episode_steps=1000


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
        "state_dim": state_dim+2,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_sizes": [int(hs) for hs in args.hidden_sizes.split(',')],
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "device": device,
    }

    if args.policy == "BSD3_EDG":
        env_beta_map = {
            'scoop_good_ball':10,
        }

        kwargs['beta'] = env_beta_map[args.env] if args.beta == 'best' else float(args.beta)
        kwargs['with_importance_sampling'] = args.imps
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs['num_noise_samples'] = args.num_noise_samples

        policy = BSD3_EDG.SD3(**kwargs)


    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models_asia_v2/{policy_file}")

    replay_buffer = BSD3_EDG_utils.ReplayBuffer(state_dim+2, action_dim,max_size=int(args.max_size))
    replay_buffer_guide = BSD3_EDG_utils.ReplayBuffer(state_dim+2, action_dim,max_size=int(args.max_size))
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
    reach = False
    last_reach = False
    # unreach=0
    for t in range(int(args.steps)):
        episode_timesteps += 1

        # select action randomly or according to policy
        if t < args.start_timesteps:
            action = (max_action - min_action) * np.random.random(env.action_space.shape) + min_action
        else:
            if not reach and np.random.rand()<args.guide_p:
                action = (
                    policy.select_action(np.append(np.array(state),np.ones(2))) 
                    + noise()
                ).clip(-max_action, max_action)
            else:
                action = (
                            policy.select_action(np.append(np.array(state),np.zeros(2))) 
                            + noise()
                        ).clip(-max_action, max_action)
                

        next_state, reward, done, _ = env.step(action) 
        reward_guide = compute_domain_reward_2b(next_state,0.05)*0.05
        
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        # save data to replay buffer
        if episode_timesteps>1 and not done_bool_last:
            replay_buffer.add(np.append(state_last,np.zeros(2)), action_last, np.append(next_state_last,np.zeros(2)), reward_last, done_bool_last,
                              np.append(state,np.zeros(2)), action, np.append(next_state,np.zeros(2)), reward,reward, done_bool)
        if done:
            replay_buffer.add(np.append(state,np.zeros(2)), action, np.append(next_state,np.zeros(2)), reward, done_bool,
                              np.append(state,np.zeros(2)), action, np.append(next_state,np.zeros(2)), -np.inf,reward, done_bool)

        last_last_reach = last_reach
        last_reach = reach
        
        if reward_guide:
            reach = True
        else:
            reach = False

        if (not last_reach) or (not last_last_reach):
            if episode_timesteps>1 and not done_bool_last:
                replay_buffer_guide.add(np.append(state_last,np.ones(2)), action_last, np.append(next_state_last,np.ones(2)), reward_guide_last, done_bool_last,
                                        np.append(state,np.ones(2)), action, np.append(next_state,np.ones(2)), reward_guide,reward_guide, done_bool)
            if done:
                replay_buffer_guide.add(np.append(state,np.ones(2)), action, np.append(next_state,np.ones(2)), reward_guide, done_bool,
                                        np.append(state,np.ones(2)), action, np.append(next_state,np.ones(2)), -np.inf,reward_guide, done_bool)
        
        
        #==========last tmp=============#
        reward_guide_last = reward_guide
        state_last = state
        action_last = action
        next_state_last =  next_state
        reward_last = reward
        done_bool_last = done_bool
        #===============================#
        state = next_state
        episode_reward += reward
        
         
        if t >= args.start_timesteps:
            policy.train(replay_buffer,replay_buffer_guide, args.batch_size)
        

        if done or (t + 1) % args.eval_freq == 0: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            writer.add_scalar('episode_reward', episode_reward, t, walltime=time.time()) 
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            writer.add_scalar('keep_time', episode_timesteps, t, walltime=time.time()) 

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            reach = False
        
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
                eval_episodes=10
                avg_reward = 0.
                for _ in range(eval_episodes):
                    state, done = env.reset(), False
                    while not done:
                        action = policy.select_action(np.append(np.array(state),np.zeros(2)))
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
                    policy.save(f"./models_asia_v2/{file_name}")
                    print("best model saved!!!")
                flag = False



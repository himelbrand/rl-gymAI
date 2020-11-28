import gym
import numpy as np
import matplotlib.pyplot as plt
import pprint
from datetime import datetime
from collections import defaultdict
DEBUG = False
PRIOR = False
CHANGING_STEPS = False
MAX_STEPS = 10000
MAP = '8x8'
png_suffix = ''
check_states = []
terminating_states = {'4x4':[15,5,7,11,12],'8x8':[63,59,54,52,49,46,42,41,35,29,19]}

def set_prior(value):
    global PRIOR
    PRIOR = value

def set_debug(value):
    global DEBUG
    DEBUG = value

def set_png_suffix(value):
    global png_suffix
    png_suffix = value

def set_max_steps(value):
    global MAX_STEPS
    MAX_STEPS = value

def set_changing_steps(value):
    global CHANGING_STEPS
    CHANGING_STEPS = value

def use_small_map(value):
    global MAP
    if value:
        MAP = '4x4'

def init_env(max_steps=250):
    if MAP == '4x4':
        env =  gym.make('FrozenLake-v0')
    elif MAP == '8x8':
        env =  gym.make('FrozenLake8x8-v0')
    env._max_episode_steps = max_steps
    return env

def modify_env(env):
    def new_reset(state=None):
        env.orig_reset()  
        if state is not None:
            env.env.s = state
        return np.array(env.env.s)

    env.orig_reset =  env.reset
    env.reset = new_reset
    return env

def draw_initial_state(env):
    while True:
        s_init = env.observation_space.sample()
        if not PRIOR or s_init not in terminating_states[MAP]:
            break
    return s_init

def evaluate(env,pi,gamma,episodes_num=500):
    if DEBUG:
        print('Running policy evaluation')
    v = np.zeros(env.observation_space.n)
    goal_reached = 0
    seen = 0
    for _ in range(episodes_num):
        s = env.reset(0)
        done = False
        steps = 0
        sample = []
        while(not done):
            if s == 0:
                seen += 1
            a = np.argmax(pi[s])
            s_tag, r, done, _ = env.step(a)
            sample.append((s,a,r))
            goal_reached += int(r>0)
            s = s_tag
            steps += 1    
        for t in range(len(sample)):
            s,a,r = sample[t]
            Gt = r
            for t_tag in range(t+1,len(sample)):
                e = t_tag - t
                s_tag,_,r_tag = sample[t_tag]
                discount = gamma**e
                Gt += r_tag*discount
            v[s] += Gt
    v0 = v[0]/seen
    if DEBUG:
        print(f'Reached goal in {goal_reached}/{episodes_num} episodes')
    return v0
    
def sarsa(env,Q,pi,gamma,Lambda,alpha,states,actions,eps,max_step=5000,episode_max_steps=250,iters=0):
    steps = 0
    if DEBUG:
        print(f'Running iteration {iters} of SARSA')
    goal_reached = 0
    episodes = 0
    epsilon_decay = 0.9999
    while steps < max_step:
        E = np.zeros(Q.shape)
        episodes += 1
        s = draw_initial_state(env)
        a = np.random.choice(actions,p=pi[s])
        env.reset(s)
        for _ in range(episode_max_steps):
            s_tag, reward, done, _ = env.step(a)
            goal_reached += int(reward>0)
            eps_greedy(eps,pi,Q,states,actions)
            a_tag = np.random.choice(actions,p=pi[s_tag])
            delta = reward + gamma*Q[s_tag][a_tag] - Q[s][a]
            E[s][a] += 1
            for s in states:
                for a in actions:
                    Q[s][a] += alpha*delta*E[s][a]
                    E[s][a] *= gamma*Lambda
            s, a = s_tag, a_tag
            steps += 1
            if done or steps >= max_step: break
        eps *= epsilon_decay
    if DEBUG:
        print(f'Reached goal {goal_reached}/{episodes} in episodes of this iteration')
    return steps,episodes,eps
            
def eps_greedy(eps,pi,q,states,actions):
    def random_argmax(a):
        return np.random.choice(np.where(a == a.max())[0])
    for s in states:
        a_star = random_argmax(q[s])
        uni = np.float(eps/len(actions))
        for a in actions: 
            pi[s][a] = uni + ((1-eps) if a == a_star else 0)

def get_maxsteps(curr):
    if CHANGING_STEPS:
        if curr < 250000:
            return MAX_STEPS
        if curr < 750000:
            return MAX_STEPS//2
        return MAX_STEPS//3
    else:
        return MAX_STEPS
   

def learn_policy(env,actions,states,gamma,Lambda,alpha):
    #init Q
    Q = np.zeros((len(states),len(actions)))
    #init pi
    pi = np.ones((len(states),len(actions)))
    pi /= len(actions) 
    x = [0]
    y = [0]
    total_steps = 0
    total_episodes = 0
    iters = 0
    epsilon = 1
    while total_steps < 1e6:
        iters += 1
        steps,episodes,epsilon = sarsa(env,Q,pi,gamma,Lambda,alpha,states,actions,epsilon,max_step=get_maxsteps(total_steps),episode_max_steps=250,iters=iters)
        v = evaluate(env,pi,gamma)
        total_episodes += episodes
        total_steps += steps
        x.append(total_steps)
        y.append(v)  
        if DEBUG:
            print(f'current step count is: {total_steps} with epsilon={epsilon}')      
    return {'x':x,'y':y},pi

def human_agent(env):
    a = -1
    while True:
        ans = input('''Make your move: 
0) Left
1) Down
2) Right
3) Up

Your action:\t''')
        try:
            a = int(ans)
            if env.action_space.contains(a):
                break
        except:
            print('Your action choice must be an integer')
        print('Try again - action not in action space')
    return a

def run_simulation(env,policy=None,human=False):
    s = env.reset()
    done = False
    score = 0
    steps = 0
    if policy is None and not human:
        print('No policy found - using human agent')
        human = True
    while(not done):
        env.render()
        a = human_agent(env) if human else np.argmax(policy[s])
        s, r, done, _ = env.step(a)
        score += r
        steps += 1
    env.render()
    print(f'Final score: {score}')
    print(f'Done in {steps} steps')
    env.close()

def main(gamma=0.95,human=False):
    print(f'Using {MAP} map')
    values = {}
    env = init_env()
    if human:
        run_simulation(env,human=True)
        return
    tmp_env = modify_env(env)
    nA = env.action_space.n
    nS = env.observation_space.n
    for Lambda in [0.9,0.7,0.5]:
        for alpha in [0.1,0.05]:
            print(f'Learning policy using lambda={Lambda} and alpha={alpha}')
            label = f'$\\alpha={alpha},\\lambda={Lambda}$'
            xy,pi = learn_policy(tmp_env,range(nA),range(nS),gamma,Lambda,alpha)
            values[label] = xy
            run_simulation(env,policy=pi)
            print(f'Done running a single simulation using learned policy with lambda={Lambda} and alpha={alpha}')
    plt.figure(figsize=(20,10))
    plt.title(r'Learning of policy - $V^{\\pi}_{init}$ by steps')
    plt.xlabel('Total steps')
    plt.ylabel(r'$V^{\\pi}_{init}$',rotation=90)
    for label in values:
        x,y = values[label]['x'],values[label]['y']
        plt.plot(x,y,label=label)
    plt.legend()
    plt.savefig(f'out/plot{MAP}({datetime.strftime(datetime.now(),"%d-%m_%H:%M")}){png_suffix}.png')
if __name__ == "__main__":
    import argparse
    def parse_args():
        global MAP
        parser = argparse.ArgumentParser(prog='hw1.py',description='AI agent using SARSA lambda for AI-Gym Frozen lake.')
        parser.add_argument('-human',dest='human', action='store_true',help='use this flag to run human agent')
        parser.add_argument('-gamma',dest='gamma', metavar='G',default=0.95, type=float, help='a float for gamma in [0,1] (default: 0.95).')
        parser.add_argument('-d',dest='debug', action='store_true',help='use this flag to get debug prints')
        parser.add_argument('-cs',dest='c_steps', action='store_true',help='use this flag to have changing step counts between evaluations')
        parser.add_argument('-ms',dest='max_steps', metavar='MAX_STEPS',default=10000, type=int, help='a int for number of steps between evaluations.')
        parser.add_argument('-png',dest='png', metavar='PNG_SUFFIX',default='', help='a suffix for png out file')
        parser.add_argument('-4x4',dest='map', action='store_true',help='use this flag to use 4x4 map')
        parser.add_argument('-p',dest='prior', action='store_true',help='use this flag to use information regarding terminating states')
        args = parser.parse_args()
        if args.gamma > 1 or args.gamma < 0:
            raise argparse.ArgumentTypeError(f'{args.gamma} must be in the interval [0,1].')
        set_debug(args.debug)
        set_prior(args.prior)
        set_changing_steps(args.c_steps)
        set_max_steps(args.max_steps)
        set_png_suffix(args.png)
        use_small_map(args.map)
        return args

    args = parse_args()    
    main(gamma=args.gamma,human=args.human)

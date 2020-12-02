import gym
import numpy as np
import matplotlib.pyplot as plt
import pprint
from datetime import datetime
from collections import defaultdict
from itertools import product  
DEBUG = False
MAX_STEPS = 1000
MAP = '8x8'
png_suffix = ''
terminating_states = {'4x4':[15,5,7,11,12],'8x8':[63,59,54,52,49,46,42,41,35,29,19]}

def set_debug(value):
    global DEBUG
    DEBUG = value

def set_png_suffix(value):
    global png_suffix
    png_suffix = value

def set_max_steps(value):
    global MAX_STEPS
    MAX_STEPS = value

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

def save_run(pi,x,y,alpha,Lambda):
    try:
        with open(f'{alpha}-{Lambda}-x.npy','wb') as f:
            np.save(f,np.array(x))
        with open(f'{alpha}-{Lambda}-y.npy','wb') as f:
            np.save(f,np.array(y))
        with open(f'{alpha}-{Lambda}-pi.npy','wb') as f:
            np.save(f,pi)
    except:
        print('failed to save V and pi to files')

def plot_results(values):
    def powerset(s):
        res = []
        x = len(s)
        for i in range(1 << x):
            res.append([s[j] for j in range(x) if (i & (1 << j))])
        return res

    sizes = [
        [1000]*5,
        [2000]*5,
        [4000]*5,
        [5000]*5,
        [8000]*5,
        [10000]*5,
        [2000,4000,5000,8000,10000],
        [1000,2000,4000,5000,8000],
        [1000,2000,3000,4000,5000],
        [1000,2500,5000,7500,10000]
        ]
    lambdas = powerset([0.9,0.7,0.5])
    alphas = powerset([0.03,0.02,0.01])
    for j,(s,l,a) in enumerate(product(sizes,lambdas,alphas)):
        plt.figure(figsize=(20,10))
        plt.title(r'Learning of policy - $V^{\pi}_{init}$ by steps')
        plt.xlabel('Total steps')
        plt.ylabel(r'$V^{\pi}_{init}$',rotation=90)
        for label in values:
            x,y,alpha,Lambda = values[label][0]['x'],values[label][0]['y'],values[label][1],values[label][2]
            if alpha in a and Lambda in l:
                y = [v for i,v in enumerate(y) if x[i] % s[x[i]//200001] == 0]
                x = [step for step in x if step % s[step//200001] == 0]
                plt.plot(x,y,label=label)
                plt.legend()
                plt.savefig(f'out/plot{MAP}({datetime.strftime(datetime.now(),"%d-%m_%H:%M")})_{min(s)}-{max(s)}{png_suffix}_{j}.png')
        plt.close()
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

def evaluate(env,pi,gamma,episodes_num=750):
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
        print(f'Reached goal in {goal_reached}/{episodes_num} episodes, got V(0)={v0}')
    return v0

def eps_greedy(eps,pi,q,states,actions):
    def random_argmax(a):
        return np.random.choice(np.where(a == a.max())[0])
    for s in states:
        a_star = random_argmax(q[s])
        uni = np.float(eps/len(actions))
        for a in actions: 
            pi[s][a] = uni + ((1-eps) if a == a_star else 0)

def sarsa(env,Q,pi,gamma,Lambda,alpha,states,actions,eps,explored,max_step=5000,episode_max_steps=250,iters=0,epsilon_decay=0.99999,min_eps=0.1):
    steps = 0
    if DEBUG:
        print(f'Running iteration {iters} of SARSA')
    goal_reached = 0
    episodes = 0
    while steps < max_step:
        E = np.zeros(Q.shape)
        episodes += 1
        s = 0
        a = np.random.choice(actions,p=pi[s])
        env.reset(s)
        for _ in range(episode_max_steps):
            s_tag, reward, done, _ = env.step(a)
            goal_reached += int(reward>0)
            eps_greedy(eps,pi,Q,states,actions)
            a_tag = np.random.choice(actions,p=pi[s_tag])
            delta = reward + gamma*Q[s_tag][a_tag] - Q[s][a]
            E[s][a] += 1
            explored[s][a] += 1
            for s in states:
                for a in actions:
                    Q[s][a] += alpha*delta*E[s][a]
                    E[s][a] *= gamma*Lambda
            s, a = s_tag, a_tag
            steps += 1
            if done or steps >= max_step: break
        if reward:
            eps *= 0.9
        eps = max(min_eps,eps*epsilon_decay) 
    if DEBUG:
        print(f'Reached goal {goal_reached}/{episodes} in episodes of this iteration')
    return steps,eps

def learn_policy(env,actions,states,gamma,Lambda,alpha):
    nS,nA = len(states),len(actions)
    shape = (nS,nA)
    #best pi for return + debugging 
    best_pi = None
    best_v0 = float('-inf')
    #E is for debugging of exploration
    E = np.zeros(shape)
    E[terminating_states[MAP]] = 1
    #init Q
    Q = np.ones(shape)
    Q[terminating_states[MAP]] = 0
    #init pi
    pi = np.ones(shape)
    pi /= nA
    x,y = [0],[0]
    total_steps = 0
    iters = 0
    epsilon = 1
    while total_steps < 1e6:
        iters += 1
        steps,epsilon = sarsa(env,Q,pi,gamma,Lambda,alpha,states,actions,epsilon,E,max_step=MAX_STEPS,iters=iters)
        v0 = evaluate(env,pi,gamma)
        if v0 >= best_v0:
            best_v0 = v0
            best_pi = pi.copy()
        total_steps += steps
        x.append(total_steps)
        y.append(v0)  
        if DEBUG:
            print(f'current step count is: {total_steps} with epsilon={epsilon}')    
            if iters%100 == 0:
                for s in states:
                    print(f'Q({s}) -> argmax({np.argmax(Q[s])}),max({np.max(Q[s])})')
                    tmp = [str(x) for x in np.where(E[s]==0)[0]]
                    if len(tmp):
                        print(f'Still need exploring: {", ".join(tmp)}')

    save_run(best_pi,x,y,alpha,Lambda)
    return {'x':x,'y':y},best_pi

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
        for alpha in [0.03,0.02,0.01]:
            print(f'Learning policy using lambda={Lambda} and alpha={alpha}')
            label = f'$\\alpha={alpha},\\lambda={Lambda}$'
            xy,pi = learn_policy(tmp_env,range(nA),range(nS),gamma,Lambda,alpha)
            values[label] = (xy,alpha,Lambda)
            run_simulation(env,policy=pi)
            print(f'Done running a single simulation using learned policy with lambda={Lambda} and alpha={alpha}')
    print('Creating tons of plots to pick the most informative from...')
    plot_results(values)
    print('All possible plots can be now found in out directory!')

if __name__ == "__main__":
    import argparse
    def parse_args():
        global MAP
        parser = argparse.ArgumentParser(prog='hw1.py',description='AI agent using SARSA lambda for AI-Gym Frozen lake.')
        parser.add_argument('-human',dest='human', action='store_true',help='use this flag to run human agent')
        parser.add_argument('-gamma',dest='gamma', metavar='G',default=0.95, type=float, help='a float for gamma in [0,1] (default: 0.95).')
        parser.add_argument('-d',dest='debug', action='store_true',help='use this flag to get debug prints')
        parser.add_argument('-ms',dest='max_steps', metavar='MAX_STEPS',default=1000, type=int, help='a int for number of steps between evaluations.')
        parser.add_argument('-png',dest='png', metavar='PNG_SUFFIX',default='', help='a suffix for png out file')
        parser.add_argument('-4x4',dest='map', action='store_true',help='use this flag to use 4x4 map')
        args = parser.parse_args()
        if args.gamma > 1 or args.gamma < 0:
            raise argparse.ArgumentTypeError(f'{args.gamma} must be in the interval [0,1].')
        set_debug(args.debug)
        set_max_steps(args.max_steps)
        set_png_suffix(args.png)
        use_small_map(args.map)
        return args

    args = parse_args()    
    main(gamma=args.gamma,human=args.human)

import gym
import numpy as np
import matplotlib.pyplot as plt
import pprint
from datetime import datetime
from collections import defaultdict
from itertools import product  
DEBUG = False
MAX_STEPS = 120000
EVAL_STEPS = 1000
RELAXED = False
png_suffix = ''
P_CENTERS = [(i)*0.18 for i in range(-4,4)]#[-0.72, -0.54, -0.36, -0.18, 0.0, 0.18, 0.36, 0.54] #[(i)*0.19 for i in range(-4,4)]
V_CENTERS = [(i)*0.014 for i in range(-4,4)]
P_I = 0.18
V_I = 0.014
CENTER_PRODUCTS = np.array(list(product(P_CENTERS,V_CENTERS)))
SIGMA_P = 0.04
SIGMA_V = 0.0004
COV = np.diag([SIGMA_P,SIGMA_V])
INV_COV = np.linalg.inv(COV)
MIN_V = -500

def set_debug(value):
    global DEBUG
    DEBUG = value

def set_relaxed(value):
    global RELAXED
    RELAXED = value

def set_png_suffix(value):
    global png_suffix
    png_suffix = value

def set_max_steps(value):
    global MAX_STEPS
    MAX_STEPS = value

def set_eval_steps(value):
    global EVAL_STEPS
    EVAL_STEPS = value


def init_env(max_steps=500):
    global MIN_V
    env =  gym.make('MountainCar-v0')
    MIN_V = -max_steps
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

def save_run(w,x,y,alpha,Lambda):
    t = 'relaxed-' if RELAXED else ''
    try:
        with open(f'out/{t}{alpha}-{Lambda}-x.npy','wb') as f:
            np.save(f,np.array(x))
        with open(f'out/{t}{alpha}-{Lambda}-y.npy','wb') as f:
            np.save(f,np.array(y))
        with open(f'out/{t}{alpha}-{Lambda}-w.npy','wb') as f:
            np.save(f,w)
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
        [3000]*5,
        [4000]*5,
        [5000]*5,
        [2000,3000,4000,5000,5000],
        [1000,2000,3000,4000,5000],
        [1000,1000,2000,3000,4000],
        [1000,1000,2000,2000,3000],
        [1000,1000,2000,3000,4000],
        [2000,2000,3000,4000,5000],
        [5000,4000,3000,2000,1000]
        ]
    lambdas = powerset([0.5])
    alphas = powerset([0.02])
    t = '_relaxed' if RELAXED else ''
    part = int(MAX_STEPS/5)+1
    for (s,l,a) in product(sizes,lambdas,alphas):
        plt.figure(figsize=(20,10))
        plt.title(r'Learning of policy - $V^{\pi}_{init}$ by steps')
        plt.xlabel('Total steps')
        plt.ylabel(r'$V^{\pi}_{init}$',rotation=90)
        for label in values:
            x,y,alpha,Lambda = values[label][0]['x'],values[label][0]['y'],values[label][1],values[label][2]
            if alpha in a and Lambda in l:
                y = [v for i,v in enumerate(y) if x[i] % s[x[i]//part] == 0]
                x = [step for step in x if step % s[step//part] == 0]
                plt.plot(x,y,label=label)
                plt.legend()
                plt.savefig(f'out/plots/plot({datetime.strftime(datetime.now(),"%d-%m_%H:%M")}){t}_{"-".join([str(n) for n in s])}{png_suffix}.png')
        plt.close()
def human_agent(env):
    from readchar import readkey
    left_arrow = '\x1b[D'
    right_arrow = '\x1b[C'
    mapping = dict({left_arrow:0,right_arrow:2})
    key = readkey()
    return mapping.get(key,1)

def run_simulation(env,w=None,human=False,show=True):
    s = env.reset()
    actions = range(env.action_space.n)
    done = False
    score = 0
    steps = 0
    if w is None and not human:
        print('No policy found - using human agent')
        human = True
    while(not done):
        if show:
            env.render()
        Qhat = QApproximation(theta(centers_distance(s)),w)
        a = human_agent(env) if human else apply_policy(Qhat,actions)
        s, r, done, _ = env.step(a)
        score += r
        steps += 1
    if show:
        env.render()
    print(f'Final score: {score}')
    print(f'Done in {steps} steps')
    env.close()
    return score

def evaluate(env,w,gamma,episodes_num=100,show=False):
    if DEBUG:
        print('Running policy evaluation')
    v0 = 0
    actions = range(env.action_space.n)
    goal_reached = 0
    seen = 0
    for i in range(episodes_num):
        s = env.reset()
        x_s = centers_distance(s)
        s0 = s
        theta_s = theta(x_s)
        Qhat = QApproximation(theta_s,w)
        done = False
        steps = 0
        sample = []
        while(not done):
            if i == 99 and show and DEBUG:
                env.render()
            if np.all(s0==s):
                seen += 1
            a = apply_policy(Qhat,actions)
            s_tag, r, done, _ = env.step(a)
            x_s_tag = centers_distance(s_tag)
            theta_s_tag = theta(x_s_tag)
            Qhat_tag = QApproximation(theta_s_tag,w)
            sample.append((s,a,r))
            goal_reached += int(goal_test(s_tag))
            s = x_s_tag
            Qhat = Qhat_tag
            steps += 1  
        if i == 99 and show and DEBUG:
            env.render()
        for t in range(len(sample)):
            s,a,r = sample[t]
            Gt = r
            for t_tag in range(t+1,len(sample)):
                e = t_tag - t
                s_tag,_,r_tag = sample[t_tag]
                discount = gamma**e
                Gt += r_tag*discount
            if np.all(s0==s):
                v0 += Gt
        if i == 99 and show and DEBUG:
            env.close()
    v0 = v0/seen
    if DEBUG:
        print(f'Reached goal in {goal_reached}/{episodes_num} episodes, got V(0)={v0}')
    return v0

def apply_policy(Qhat,actions,eps=0):
    flip = np.random.rand()
    if flip < eps:
        return np.random.choice(actions)
    else:
        return np.argmax(Qhat)

def theta(x:np.ndarray):
    return np.exp(-(np.dot(x,INV_COV)*x).sum(axis=1)/2)

def centers_distance(s:np.ndarray):
    return s-CENTER_PRODUCTS
   
def tiles(x:np.ndarray):
    p_min = min([abs(d[0]) for d in x])
    v_min = min([abs(d[1]) for d in x])
    y = [int(abs(d[0]) == p_min and abs(d[1]) == v_min) for d in x]
    return np.array(y)

def QApproximation(theta:np.ndarray,w:np.ndarray):
    return (theta*w).sum(axis=1)

def init_weights(nA=3,seed=27021990):
    if DEBUG:
        np.random.seed(seed=seed)
    return np.random.randn(nA,len(CENTER_PRODUCTS)) 

def goal_test(s:np.ndarray):
    p,v = s
    return p >= 0.5 and v >= 0

def state2key(s:np.ndarray):
    return s.tobytes()

def sarsa(env,w,gamma,Lambda,alpha,actions,eps,max_step=5000,iters=0,epsilon_decay=0.99,min_eps=0.1):
    steps = 0
    if DEBUG:
        print(f'Running iteration {iters} of SARSA')
    goal_reached = 0
    episodes = 0
    while steps < max_step:
        episodes += 1
        s = env.reset()
        x_s = centers_distance(s)
        theta_s = theta(x_s)
        Qhat = QApproximation(theta_s,w)
        E = np.zeros((len(actions),len(theta_s)))
        a = apply_policy(Qhat,actions,eps=eps)
        done = False
        while not done:
            s_tag, reward, done, _ = env.step(a)
            x_s_tag = centers_distance(s_tag)
            theta_s_tag = theta(x_s_tag)
            Qhat_tag = QApproximation(theta_s_tag,w)
            goal = int(goal_test(s_tag))
            a_tag = apply_policy(Qhat_tag,actions,eps=eps)
            delta = reward - Qhat[a] + gamma*Qhat_tag[a] if not done else 0
            E[a] += tiles(x_s)
            w[a] += alpha*delta*E[a]*theta_s
            E *= gamma*Lambda 
            s, a = s_tag, a_tag
            x_s = x_s_tag
            theta_s = theta_s_tag
            Qhat = Qhat_tag
            steps += 1
            if steps >= max_step:
                break
        if goal:
            eps *= 0.99
        goal_reached += goal
        eps = max(min_eps,eps*epsilon_decay) 
    if DEBUG:
        print(f'Reached goal {goal_reached}/{episodes} in episodes of this iteration')
    return steps,episodes,eps

def learn_policy(env,actions,gamma,Lambda,alpha):
    nA = len(actions)
    #best pi for return + debugging 
    best_w = None
    best_v0 = float('-inf')
    best_time = (0,0)
    #init weights
    w = init_weights(nA=nA)
    x,y = [0],[MIN_V]
    total_steps = 0
    total_episodes = 0
    iters = 0
    epsilon = 1
    while total_steps < MAX_STEPS:
        iters += 1
        steps,episodes,epsilon = sarsa(env,w,gamma,Lambda,alpha,actions,epsilon,max_step=EVAL_STEPS,iters=iters)
        v0 = evaluate(env,w,gamma,show=iters%50==0)
        total_steps += steps
        total_episodes += episodes
        if v0 >= best_v0:
            best_v0 = v0
            best_w = w.copy()
            best_time = (total_steps,total_episodes)
        x.append(total_steps)
        y.append(v0)  
        if DEBUG:
            print(f'current step count is: {total_steps} with epsilon={epsilon}')    

    save_run(best_w,x,y,alpha,Lambda)
    return {'x':x,'y':y},best_w,best_v0,best_time

def main(gamma=1,human=False):
    values = {}
    env = init_env(max_steps=500 if RELAXED else 200)
    if human:
        run_simulation(env,human=True)
        return
    try:
        with open(f'out/{"relaxed-" if RELAXED else ""}0.02-0.5-w.npy','rb') as f:
            w = np.load(f)
            print('Running simulation using previous learned policy')
            run_simulation(env,w=w)
    except:
        print('No previous learned policy found...')
    nA = env.action_space.n
    for Lambda in [0.5]:
        for alpha in [0.02]:
            print(f'Learning policy using lambda={Lambda} and alpha={alpha}')
            label = f'$\\alpha={alpha},\\lambda={Lambda}$'
            xy,w,v0,(steps,episodes) = learn_policy(env,range(nA),gamma,Lambda,alpha)
            print(f'The best policy was learned in {episodes} episodes or in {steps} steps and V(0)={v0}')
            values[label] = (xy,alpha,Lambda)
            run_simulation(env,w=w)
            print(f'Done running a single simulation using learned policy with lambda={Lambda} and alpha={alpha}')
    print('Creating tons of plots to pick the most informative from...')
    plot_results(values)
    print('All possible plots can be now found in out directory!')

if __name__ == "__main__":
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(prog='hw2.py',description='AI agent using SARSA lambda for AI-Gym Frozen lake.')
        parser.add_argument('-human',dest='human', action='store_true',help='use this flag to run human agent')
        parser.add_argument('-gamma',dest='gamma', metavar='G',default=1.0, type=float, help='a float for gamma in [0,1] (default: 0.95).')
        parser.add_argument('-d',dest='debug', action='store_true',help='use this flag to get debug prints')
        parser.add_argument('-ms',dest='max_steps', metavar='MAX_STEPS',default=120000, type=int, help='a int for number of maximum steps for learning.')
        parser.add_argument('-es',dest='eval_steps', metavar='EVAL_STEPS',default=1000, type=int, help='a int for number of steps between evaluations.')
        parser.add_argument('-png',dest='png', metavar='PNG_SUFFIX',default='', help='a suffix for png out file')
        parser.add_argument('-relax',action='store_true',help='use this flag to use 500 steps episodes')
        args = parser.parse_args()
        if args.gamma > 1 or args.gamma < 0:
            raise argparse.ArgumentTypeError(f'{args.gamma} must be in the interval [0,1].')
        set_debug(args.debug)
        set_max_steps(args.max_steps)
        set_eval_steps(args.eval_steps)
        set_relaxed(args.relax)
        set_png_suffix(args.png)
        return args

    args = parse_args()    
    main(gamma=args.gamma,human=args.human)

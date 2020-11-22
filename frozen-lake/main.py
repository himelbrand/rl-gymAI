import gym
import numpy as np
import matplotlib.pyplot as plt
import pprint
from datetime import datetime
DEBUG = False
PRIOR = False
MAP = '8x8'
check_states = []
terminating_states = {16:[15,5,7,11,12],64:[63,59,54,52,49,46,42,41,35,29,19]}

def set_prior(value):
    global PRIOR
    PRIOR = value

def set_debug(value):
    global DEBUG
    DEBUG = value

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
        if PRIOR and s_init not in terminating_states[env.observation_space.n]:
            break
        elif not PRIOR:
            break
    return s_init

def evaluate(env,pi,gamma):
    if DEBUG:
        print('Running policy evaluation')
    v = np.zeros(env.observation_space.n)
    goal_reached = 0
    for _ in range(15):
        s = env.reset(draw_initial_state(env))
        done = False
        steps = 0
        sample = []
        while(not done):
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
    v = v/15
    if DEBUG:
        print(f'Reached goal in {goal_reached}/15 episodes')
    return v
    
def sarsa(env,Q,pi,gamma,Lambda,alpha,states,actions,eps,max_step=5000,episode_max_steps=250,iters=0):
    steps = 0
    if DEBUG:
        print(f'Running iteration {iters} of SARSA')
    goal_reached = 0
    episodes = 0
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
        eps_greedy(eps,pi,Q,states,actions)
    if DEBUG:
        print(f'Reached goal {goal_reached}/{episodes} in episodes of this iteration')
        if goal_reached/episodes > 0.9:
            print(f'\n\n{"$"*50}{goal_reached/episodes}{"$"*50}\n\n')
    return steps,episodes
            
def eps_greedy(eps,pi,q,states,actions):
    for s in states:
        a_star = np.argmax(q[s])
        uni = np.float(eps/len(actions))
        for a in actions: 
            pi[s][a] = uni + ((1-eps) if a == a_star else 0)

def check_pi_convergence(pi,og_pi):
    for s,actions in enumerate(pi):
        a = np.argmax(actions)
        t = np.argmin(actions)
        b = np.argmax(og_pi[s])
        if b != a or a==t:
            return False
    return True

def check_Q_convergence(Q,og_Q,actions):
    for s,values in enumerate(Q):
        order_a = np.array(list(sorted(actions,key=lambda x:values[x])))
        order_b = np.array(list(sorted(actions,key=lambda x:og_Q[s][x])))
        if not (order_a==order_b).all():
            return False
    return True

def get_maxsteps(curr):#maybe try and change number of steps between evaluations
    return 2500
    # if curr < 20000:
    #     return 1000
    # if curr < 60000:
    #     return 2000
    # return 3000
   

def learn_policy(env,actions,states,gamma,Lambda,alpha):
    #init Q
    Q = np.zeros((len(states),len(actions)))
    #init pi
    pi = np.ones((len(states),len(actions)))
    pi /= len(actions) 
    x = []
    y = []
    z = []
    total_steps = 0
    total_episodes = 0
    iters = 0
    last_iter = False
    while True:
        iters += 1
        epsilon = 1/iters #maybe should be different epsilon
        if not last_iter:
            og_pi = pi.copy()
            og_Q = Q.copy()
        steps,episodes = sarsa(env,Q,pi,gamma,Lambda,alpha,states,actions,epsilon,max_step=get_maxsteps(total_steps),episode_max_steps=250,iters=iters)
        v = evaluate(env,pi,gamma)
        total_episodes += episodes
        total_steps += steps
        x.append(total_steps)
        y.append(v[0])
        z.append(v.mean())
        if check_Q_convergence(Q,og_Q,actions) and check_pi_convergence(pi,og_pi):#checking that ordering stayed the same
            if last_iter:#giving another chance before deciding we got converged
                break
            last_iter = True
        else:
            last_iter = False
    
    #show plot
    plt.plot(x,y,label='V(0)')
    plt.plot(x,z,label='mean(V)')
    plt.xlabel('time steps')
    plt.ylabel('policy value')
    plt.legend()
    plt.title(f'Learning plot for $\\alpha$={alpha} and $\\lambda$={Lambda} with{" " if PRIOR else "out "} prior')
    # plt.show()
    plt.savefig(f'out/{MAP}/plot-{Lambda}_a-{alpha}_{datetime.now().strftime("%d-%m_%H:%M:%S")}.png')
    plt.close()
    return pi,total_episodes,total_steps

def human_agent(env):
    a = -1
    while True:
        ans = input('''make your move: 
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

def main(gamma=0.95,save=True,load=None,human=False):
    print(f'Using {MAP} map')
    env = init_env()
    if human:
        run_simulation(env,human=True)
        return
    if load is not None:
        with open(load, 'rb') as f:
            pi = np.load(f) 
            run_simulation(env,policy=pi)
            return
    tmp_env = modify_env(env)
    nA = env.action_space.n
    nS = env.observation_space.n
    for Lambda in [0.95,0.9,0.7,0.5]:
        for alpha in [0.02,0.05,0.07]:
            print(f'Learning policy using lambda={Lambda} and alpha={alpha}')
            pi,episodes,steps = learn_policy(tmp_env,range(nA),range(nS),gamma,Lambda,alpha)
            
            if save:
                with open(f'out/{MAP}/pi_l-{Lambda}_a-{alpha}_{datetime.now().strftime("%d-%m_%H:%M:%S")}({"prior" if PRIOR else "no_prior"}).npy','wb') as f:
                    np.save(f,pi)
            print(f'Running a single simulation using learned policy with lambda={Lambda} and alpha={alpha}')
            run_simulation(env,policy=pi)
            print(f'Finished learning policy after {episodes} episodes and {steps} steps')

if __name__ == "__main__":
    import argparse
    def parse_args():
        global MAP
        parser = argparse.ArgumentParser(prog='hw1.py',description='AI agent using SARSA lambda for AI-Gym Frozen lake.')
        parser.add_argument('-human',dest='human', action='store_true',help='use this flag to run human agent')
        parser.add_argument('-gamma',dest='gamma', metavar='G',default=0.95, type=float, help='a float for gamma in [0,1] (default: 0.95).')
        parser.add_argument('-d',dest='debug', action='store_true',help='use this flag to get debug prints')
        parser.add_argument('-4x4',dest='map', action='store_true',help='use this flag to use 4x4 map')
        parser.add_argument('-p',dest='prior', action='store_true',help='use this flag to use information regarding terminating states')
        parser.add_argument('-save',dest='save', action='store_true',help='use this flag to save pi to file')
        parser.add_argument('-load',dest='load',metavar='FILE',help='use this flag to load pi from file',default=None)
        args = parser.parse_args()
        if args.gamma > 1 or args.gamma < 0:
            raise argparse.ArgumentTypeError(f'{args.gamma} must be in the interval [0,1].')
        set_debug(args.debug)
        set_prior(args.prior)
        use_small_map(args.map)
        return args

    args = parse_args()
    main(gamma=args.gamma,save=args.save,load=args.load,human=args.human)

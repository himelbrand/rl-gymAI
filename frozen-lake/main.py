import gym
import numpy as np

DEBUG = False
PRIOR = False
check_states = []


def set_prior(value):
    global PRIOR
    PRIOR = value

def set_debug(value):
    global DEBUG
    DEBUG = value

def init_env():
    return gym.make('FrozenLake8x8-v0')

def modify_env(env):
    def new_reset(state=None):
        env.orig_reset()  
        if state is not None:
            env.env.s = state
        return np.array(env.env.s)

    env.orig_reset =  env.reset
    env.reset = new_reset
    return env

def explore_env(env):
    tmp_env = modify_env(env)
    P = []
    nS,nA = 0,0
    # compute number of states
    while True:
        tmp_env.reset(nS)
        try:
            tmp_env.step(0)
            nS += 1
        except:
            break
    # compute number of actions
    while True:
        tmp_env.reset(0)
        try:
            tmp_env.step(nA)
            nA += 1
        except:
            break
    for s in range(nS):
        P_s = []
        for a in range(nA):
            P_s_a = []
            prob_sum = 0
            while prob_sum < 1:
                tmp_env.reset(s)
                s_tag, reward, done, info = tmp_env.step(a)
                prob_sum += info['prob']
                P_s_a.append({'next_s':s_tag,'done':done,'reward':reward,'p':info['prob']})
            P_s.append(P_s_a)
        P.append(P_s)
    return P, nS, nA

def init(P,states,actions):
    nS,nA = len(states),len(actions)
    pi_0 = np.zeros((nS,nA))
    R = np.zeros((nS,nA))
    v = np.zeros(nS)
    goals = set()
    for s in states:
        valid_actions = []
        for a in actions:
            for T in P[s][a]:
                if PRIOR and T['next_s'] != s and T['reward'] >= -1:
                    valid_actions.append(a)
                R[s][a] = T['reward']
                if T['done']:
                    goals.add(T['next_s'])
        pi_0[s][np.random.choice(valid_actions if PRIOR else actions)] = 1
    return pi_0,v,R,goals

def evaluate():
    pass

def improve():
    pass

def iterate_policy():
    pass
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
        print(f'current score: {score}')
    print(f'Done in {steps} steps')
    env.close()

def main():
    env = init_env()
    run_simulation(env)#running human agent

if __name__ == "__main__":
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(prog='hw1.py',description='AI agent using policy iteration for AI-Gym Taxi.')
        parser.add_argument('-human',dest='human', action='store_true',help='use this flag to run human agent')
        parser.add_argument('-gamma',dest='gamma', metavar='G',default=0.95, type=float, help='a float for gamma in [0,1] (default: 0.95).')
        parser.add_argument('-d',dest='debug', action='store_true',help='use this flag to get debug prints')
        parser.add_argument('-p',dest='prior', action='store_true',help='use this flag to invalidate some actions from specific states and keeping all goal states with value of 0. (reduce the number of iterations needed to converge)')
        parser.add_argument('-save',dest='save', action='store_true',help='use this flag to save pi and v to files')
        parser.add_argument('-load',dest='load', action='store_true',help='use this flag to load pi and v from files')
        args = parser.parse_args()
        if args.gamma > 1 or args.gamma < 0:
            raise argparse.ArgumentTypeError(f'{args.gamma} must be in the interval [0,1].')
        set_debug(args.debug)
        set_prior(args.prior)
        return args

    args = parse_args()
    main()

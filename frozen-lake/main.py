import gym
import numpy as np
import matplotlib.pyplot as plt

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
    return gym.make('FrozenLake-v0')

def modify_env(env):
    def new_reset(state=None):
        env.orig_reset()  
        if state is not None:
            env.env.s = state
        return np.array(env.env.s)

    env.orig_reset =  env.reset
    env.reset = new_reset
    return env

# def explore_env(env):
#     tmp_env = modify_env(env)
#     P = []
#     nS,nA = 0,0
#     # compute number of states
#     while True:
#         tmp_env.reset(nS)
#         try:
#             tmp_env.step(0)
#             nS += 1
#         except:
#             break
#     # compute number of actions
#     while True:
#         tmp_env.reset(0)
#         try:
#             tmp_env.step(nA)
#             nA += 1
#         except:
#             break
#     for s in range(nS):
#         P_s = []
#         for a in range(nA):
#             P_s_a = []
#             prob_sum = 0
#             while prob_sum < 1:
#                 tmp_env.reset(s)
#                 s_tag, reward, done, info = tmp_env.step(a)
#                 prob_sum += info['prob']
#                 P_s_a.append({'next_s':s_tag,'done':done,'reward':reward,'p':info['prob']})
#             P_s.append(P_s_a)
#         P.append(P_s)
#     return P, nS, nA

# def init(P,states,actions):
#     nS,nA = len(states),len(actions)
#     pi_0 = np.zeros((nS,nA))
#     R = np.zeros((nS,nA))
#     v = np.zeros(nS)
#     goals = set()
#     for s in states:
#         valid_actions = []
#         for a in actions:
#             for T in P[s][a]:
#                 if PRIOR and T['next_s'] != s and T['reward'] >= -1:
#                     valid_actions.append(a)
#                 R[s][a] = T['reward']
#                 if T['done']:
#                     goals.add(T['next_s'])
#         pi_0[s][np.random.choice(valid_actions if PRIOR else actions)] = 1
#     return pi_0,v,R,goals

def evaluate(env,pi,gamma):
    print('running policy evaluation')
    v = np.zeros(len(pi))
    goal_reached = 0
    for episode in range(15):
        while True:
            s_init = env.observation_space.sample()#0## if episode < 14 else 0
            if s_init not in {15,5,7,11,12}:
                break
        s = env.reset(s_init)
        done = False
        steps = 0
        sample = []
        # print(f"{'='*25}episode {episode}{'='*25}")
        while(not done):
            # env.render()
            a = np.argmax(pi[s])
            s_tag, r, done, _ = env.step(a)
            sample.append((s,a,r))
            goal_reached += int(r>0)
            s = s_tag
            steps += 1
        # env.render()
        for t in range(len(sample)):
            s,a,r = sample[t]
            Gt = r
            for t_tag in range(t+1,len(sample)):
                e = t_tag - t
                s_tag,a_tag,r_tag = sample[t_tag]
                discount = gamma**e
                Gt += r_tag*discount
            print(f'state={s} with Gt={Gt}')
            v[s] += Gt
    v = v/15
    print(f'reached goal {goal_reached}/15')
    return v
    
    
def sarsa(env,Q,pi,gamma,Lambda,alpha,states,actions,eps=0.05,max_step=5000,episode_max_steps=250):
    steps = 0
    print('running SARSA')
    episodes = 0
    goal_reached = 0
    while steps < max_step:
        episodes += 1
        E = np.zeros(Q.shape)
        s = env.observation_space.sample()
        a = np.random.choice(actions,p=pi[s])
        env.reset(s)
        # print('running new episode')
        # print('Q:')
        # print(Q)
        og_pi = pi.copy()
        # eps = 1/episodes
        for _ in range(episode_max_steps):
            not_converged = 0
            s_tag, reward, done, _ = env.step(a)
            goal_reached += int(reward>0)
            pi_converged = eps_greedy(eps,pi,Q,states,actions)
            a_tag = np.random.choice(actions,p=pi[s_tag])
            delta = reward + gamma*Q[s_tag][a_tag] - Q[s][a]
            E[s][a] += 1
            for s in states:
                for a in actions:
                    not_converged += int(alpha*delta*E[s][a] != 0)
                    Q[s][a] += alpha*delta*E[s][a]
                    E[s][a] *= gamma*Lambda
            s, a = s_tag, a_tag
            steps += 1
            if done or steps >= max_step: break
        pi_converged = eps_greedy(eps,pi,Q,states,actions)
        # if reward ==1:
        #     print('Q:')
        #     print(Q)
    print(f'reached goal {goal_reached} in SARSA')
    return steps,not_converged,(pi == og_pi).all(),episodes
            


def eps_greedy(eps,pi,q,states,actions):
    converged = 0
    for s in states:
        a_star = np.argmax(q[s])
        uni = np.float(eps/len(actions))
        for a in actions:
            pi_s_a = uni + ((1-eps) if a == a_star else 0)
            converged += int(pi_s_a == pi[s][a])
            pi[s][a] = pi_s_a
    return converged == len(actions)*len(states)

def learn_policy(env,actions,states,gamma,Lambda,alpha,epsilon):
    #init Q
    Q = np.zeros((len(states),len(actions)))#np.random.rand(len(states),len(actions))
    #init pi
    pi = np.ones((len(states),len(actions)))
    pi /= len(actions) 
    x = []
    y = []
    total_steps = 0
    v_prev = None
    iters = 0
    episodes = 1
    for _ in range(50):
        iters += 1
        epsilon = 1/iters
        steps,Q_not_converged,pi_converged,episodes = sarsa(env,Q,pi,gamma,Lambda,alpha,states,actions,epsilon,max_step=2500,episode_max_steps=250)
        v = evaluate(env,pi,gamma)
        print(f'episodes done: {episodes}')
        print(Q_not_converged,pi_converged)
        if v_prev is not None:
            print(v-v_prev)
        total_steps += steps
        x.append(total_steps)
        y.append(v[0])
        if Q_not_converged == 0 and pi_converged:
            break
        v_prev = np.copy(v)
    print('V:')
    print(v)
    print('Q:')
    print(Q)
    print('pi:')
    print(pi)
    #show plot
    plt.plot(x,y)
    plt.xlabel('time steps')
    plt.ylabel('v(s_init)')
    plt.show()
    return pi

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
        # env.render()
        a = human_agent(env) if human else np.argmax(policy[s])
        s, r, done, _ = env.step(a)
        # print(s)
        score += r
        steps += 1
    print(f'current score: {score}')
    # env.render()
    print(f'Done in {steps} steps')
    env.close()

def main(gamma=0.95,epsilon=0.01):
    env = init_env()
    # env.reset()
    # env.render()
    # return
    # for s in env.P:
    #     print('='*50)
    #     for a in env.P[s]:
    #         print(f'state:{s}, action:{a}')
    #         print(env.P[s][a])
    #     print('='*50)
    # return
    tmp_env = modify_env(env)
    nA = env.action_space.n
    nS = env.observation_space.n
    for Lambda in [0.95]:
        for alpha in [0.05]:
            pi = learn_policy(tmp_env,range(nA),range(nS),gamma,Lambda,alpha,epsilon)
            run_simulation(env,policy=pi)#running human agent

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

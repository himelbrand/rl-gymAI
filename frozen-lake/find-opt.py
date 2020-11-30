import gym
import numpy as np
import pprint
env =  gym.make('FrozenLake8x8-v0')
holes = [59,54,52,49,46,42,41,35,29,19]
states = range(64)
good_actions = {}
prior = {0: ([3], True),
 1: ([3], True),
 2: ([3], True),
 3: ([3], True),
 4: ([3], True),
 5: ([3], True),
 6: ([3], True),
 7: ([2], True),
 8: ([3], True),
 9: ([3], True),
 10: ([3], True),
 11: ([3], True),
 12: ([3], True),
 13: ([2], True),
 14: ([2], True),
 15: ([2], True),
 16: ([0], True),
 17: ([0], True),
 18: ([0], True),
 20: ([2], True),
 21: ([3], True),
 22: ([2], True),
 23: ([2], True),
 24: ([0, 1, 2, 3], True),
 25: ([0, 1, 2, 3], True),
 26: ([0, 3], False),
 27: ([1, 3], False),
 28: ([0], False),
 30: ([2], True),
 31: ([2], True),
 32: ([0], True),
 33: ([3], False),
 34: ([0, 3], False),
 36: ([2], True),
 37: ([1], True),
 38: ([3], True),
 39: ([2], True),
 40: ([0], True),
 43: ([1, 2], False),
 44: ([3], False),
 45: ([0], False),
 47: ([2], True),
 48: ([0], True),
 50: ([1, 2], False),
 51: ([0, 3], False),
 53: ([0, 2], False),
 55: ([2], True),
 56: ([0, 1, 2, 3], True),
 57: ([1], True),
 58: ([0], False),
 60: ([1, 2], False),
 61: ([2], False),
 62: ([1], True)}



risky = [27, 34, 43, 50, 51, 53, 60]
best_det = [s for s in prior if prior[s][1] and len(prior[s][0])==1]
# pprint.pprint(env.P[37][1])
for s in range(7,56,8):
    print(s)
    pprint.pprint(env.P[s])
    
exit(0)
for s in states:
    valid_actions = []
    if s in holes or s == 63:
        continue
    best = []
    best_len = float('inf')
    for a in env.P[s]:
        valid = sum([3 if t[1] in holes else 1 if t[1] in risky else 0 for t in env.P[s][a]])
        if valid == 0:
            valid_actions.append(a)
        elif valid < best_len:
            best_len = valid
            best = [a]
        elif valid == best_len:
            best.append(a)
    good_actions[s] = (valid_actions,True) if len(valid_actions) else (best,False)
pprint.pprint(good_actions)
pi = np.zeros((64,4))

path = [(63,0)]
visited = [63]
fount_path = False
while not fount_path:
    last = len(path)
    for s in states:
        valid_actions = []
        if s in holes or s == 63:
            continue
        best = []
        best_len = float('inf')
        for a in env.P[s]:
            T = [t[1] for t in env.P[s][a] if t[1] != s or s not in visited]
            print(path[-1][0])
            print(T)
            if path[-1][0] in T:
                path.append((s,a))
                visited.append(s)
                if s == 0:
                    fount_path = True
                break
        if last != len(path):
            break        
    print(len(path))
    if last == len(path):
        break
print(path)
       
# for s in range(64):
#     if s in holes or s == 63:
#         a = 0
#     else:
#         a = prior[s][0][0]
#     pi[s][a] = 1
# pprint.pprint(pi)
# score = 0
# steps = 0
# for _ in range(100):
#     s = env.reset()
#     done = False
#     while not done:
#         a = np.argmax(pi[s])
#         env.render()
#         s,r,done,_ = env.step(a)
#         score += r
#         steps += 1
#     env.render()
# print(f'{score} - {score/steps}')
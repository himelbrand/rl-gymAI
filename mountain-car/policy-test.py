import hw3
import numpy as np
env = hw3.init_env(max_steps=200)
print('Testing original problem learned policy')
try:
    with open('out/0.02-0.5-w.npy','rb') as f:
        w = np.load(f)
        print('Running simulation using previous learned policy')
    wins = 0
    scores = 0 
    for i in range(100):
        score = hw3.run_simulation(env,w=w,show=i%10==0)
        wins += int(score > -110)
        scores += score
    print(f'wins {wins} out of 100')
    print(f'Policy got avreage score of {scores/100}')
    print('Win in run is defined as score higher than -110')
    print('Win in problem is defined as average score from 100 straight runs is higher than -110')
    print('\n\n\nPolicy is a Winner!' if scores/100 > -110 else '\n\n\nPolicy is a Loser!')
except:
    print('No previous learned policy found...')

print('\nTesting relaxed (500 step limit) problem learned policy')
try:
    with open('out/relaxed-0.02-0.5-w.npy','rb') as f:
        w = np.load(f)
        print('Running simulation using previous learned policy')
    wins = 0
    scores = 0 
    for i in range(100):
        score = hw3.run_simulation(env,w=w,show=i%10==0)
        wins += int(score > -110)
        scores += score
    print(f'wins {wins} out of 100')
    print(f'Policy got avreage score of {scores/100}')
    print('Win in run is defined as score higher than -110')
    print('Win in problem is defined as average score from 100 straight runs is higher than -110')
    print('\n\n\nPolicy is a Winner!' if scores/100 > -110 else '\n\n\nPolicy is a Loser!')
except:
    print('No previous learned policy found...')
    
   
python3 dqn_run.py --train ../config/train.yaml --env 'LunarLander-v2' --num_episodes 10
python3 dqn_run.py --run ../model/model.pth --env 'LunarLander-v2' --num_episodes 10
python3 dqn_run.py --tune ../config/tune.yaml --env 'LunarLander-v2' --num_episodes 2000

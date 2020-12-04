import sys
import json
import gym
import argparse
from train import train_agent
from framestack import FrameStack

def main(args):
    """ """
    with open (args.param, "r") as f:
        param = json.load(f)

    print("use the env {} ".format(param["env_name"]))
    print(param)
    print("Start Programm in {}  mode".format(args.mode))
    env = gym.make(param["env_name"])
    if args.mode == "args": 
        param["lr"] = args.lr
        param["fc1_units"] = args.fc1_units
        param["fc2_units"] = args.fc2_units
    env  = FrameStack(env, param)
    train_agent(env, param)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="hypersearch", type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--fc1_units', default=256, type=int)
    parser.add_argument('--fc2_units', default=256, type=int)
    parser.add_argument('--mode', default="pretrain", type=str)
    arg = parser.parse_args()
    main(arg)

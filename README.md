# The role of $\gamma$ in Offline RL

This is a pytorch implementation of discount factor in Offline RL on [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl), the corresponding paper is [On the Role of Discount Factor in Offline Reinforcement Learning](https://proceedings.mlr.press/v162/hu22d.html).

![Framwork](figure.png)

## Quick Start
For experiments on D4RL, our code is implemented based on TD3+BC, please click on the folder TD3+BC_gammma and then:

```shell
$ python3 main.py
```

For experiments on Toy Example, please click on the folder Toy and then:

```shell
$ python3 bcq.py
```

```shell
$ python3 dqn.py
```

## Citing
If you find this open source release useful, please reference in your paper (it is our honor):
```
@inproceedings{hu2022role,
  title={On the role of discount factor in offline reinforcement learning},
  author={Hu, Hao and Yang, Yiqin and Zhao, Qianchuan and Zhang, Chongjie},
  booktitle={International Conference on Machine Learning},
  pages={9072--9098},
  year={2022},
  organization={PMLR}
}
```

## Note
+ If you have any questions, please contact me: yangyiqi19@mails.tsinghua.edu.cn. 


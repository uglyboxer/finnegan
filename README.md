# finnegan
## by Cole Howard

An extensible custom neural net with a serious nod to [I Am Trask](http://iamtrask.github.io/2015/07/12/basic-python-network/)

Documentation, Test Coverage, and Travis
TBA

[![Coverage Status](https://coveralls.io/repos/uglyboxer/finnegan/badge.svg?branch=master&service=github)](https://coveralls.io/github/uglyboxer/finnegan?branch=master)

## To Get Started

Copy the contents of finnegan (the main repo) and from inside that folder run the command:

```
$ python3 finnegan/net_launch.py
```

To adjust hyper-parameters or choose dataset:
Open net_launch.py and adjust as needed at the bottom of the file.  The dataset can be switched by commenting out the appropriate line at the bottom and uncommneting the other.  Or simply:

```
from Network import network
```

And feed it the appropriate parameters.

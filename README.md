Remy: TCP ex Machina (computer-generated congestion control)
============================================================

This is a forked version of [Remy](https://github.com/tcpexmachina/remy) 
which adds support for reinforcement learning 
(using code from the repository [async_deep_reinforce](https://github.com/miyosuda/async_deep_reinforce)). 
The results were published in *Bachl, Maximilian, Tanja Zseby, and Joachim Fabini. "Rax: Deep Reinforcement Learning for Congestion Control." ICC 2019-2019 IEEE International Conference on Communications (ICC). IEEE, 2019.*

All files with *unicorn* in the file name were added for this purpose.

The license for Remy can be found at *debian/copyright*. All added *unicorn* files are licensed under the same license. 
The license for *async_deep_reinforce* can be found at *async_deep_reinforce/LICENSE.txt*.

The modified version `remy-unicorn` can be launched like `remy` as follows:

    learning_rate=-4.5 reward_type=no_cutoff src/remy-unicorn cf=config/2_2_really_small_buffer_0.cfg
    
This launches a simulation with 2 senders, an RTT of 100 ms, a buffer of one tenth of the bandwidth delay product, a throughput/packet-loss-tradeoff of 2, a stochastic packet loss probability of 0 and a link speed of 20 Mbit/s. 

-- Maximilian Bachl

Remy is an optimization tool to develop new TCP congestion-control
schemes, given prior knowledge about the network it will encounter
and an objective to optimize for.

It is described further at the Web site for [TCP ex
Machina](http://web.mit.edu/remy). A research paper on Remy was
published at the ACM SIGCOMM 2013 annual conference.

Basic usage:

* Remy requires a C++11 compiler to compile, e.g. gcc 4.6 or
  contemporary clang++. You will also need the Google
  protobuf-compiler and the Boost C++ libraries.

* From the version-control repository checkout, run `./autogen.sh`,
  `./configure`, and `make` to build Remy.

* Run `./remy` to design a RemyCC (congestion-control algorithm) for
  the default scenario, with link speed drawn uniformly between 10 and
  20 Mbps, minRTT drawn uniformly between 100 and 200 ms, the maximum
  degree of multiplexing drawn uniformly between 1 and 32, and each
  sender "on" for an exponentially-distributed amount of time (mean 5
  s) and off for durations drawn from the same distribution.

* Use the of= argument to have Remy save its RemyCCs to disk. It will
  save every step of the iteration.

* Use the if= argument to get Remy to read previous RemyCCs as the
  starting point for optimization.

* The `sender-runner` tool will execute saved RemyCCs. The filename
  should be set with a `if=` argument. It also accepts `link=` to set
  the link speed (in packets per millisecond), `rtt=` to set the RTT,
  and `nsrc=` to set the maximum degree of multiplexing.

If you have any questions regarding Remy, please visit [Remy's Web
site](http://web.mit.edu/remy) or e-mail `remy at mit dot edu`.

-- Keith Winstein

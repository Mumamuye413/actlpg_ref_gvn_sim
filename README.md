# Safe Autonomous Robot Navigation in Dynamic Environments 
#### This repository contains code for a demonstration of safe robot navigation in a dynamic environment using Reference Governor and Control Barrier Function techniques. It is part of Zhuolin Niu's [master thesis](https://escholarship.org/content/qt1jd778fm/qt1jd778fm.pdf).

#### In this simulation code, the differential drive robot dynamics was simulated with an unicycle model, and multiple moving obstacles are set to circle sets moving back-and-forth with a constant speed along their given trajectories. 

## Dependencies
The code is tested on `Ubuntu 20.04 LTS` with `Anaconda (Python 3.9)` 

### Libraries
- cvxpy                     1.3.0
- matplotlib                3.7.1
- numpy                     1.24.2
- scipy                     1.9.1

## Run Demo
Run the demo with `actlpg_nav_demo.py`.

- Pick `map_size` among "small", "medium" or "large";
- Pick `controller_type` between "Cone" - goal position tracking and "Polar" - goal pose alignment;
- Set `bi_direction` to `True` - bi-directional motion or `False` - forward motion only;
- Pick `save_fig` among "none" - no figure saved, "video" - save frames to make video/gif, or "time" - save figures at specific timestamps.

## Simulation Results with Different Environment Map
  * Small 1d map
    
    run the demo with `map_size="small"`
    
    <img src="/gif/sim1_1d_cone.gif" alt="1d_cone" width="600"/>
    
  * Medium 2d map with fewer(two) moving obstacles

    run the demo with `map_size="medium"`

    <img src="/gif/sim2_2o_cone.gif" alt="2o_cone" width="600"/>

  * Medium 2d map with more(eight) moving obstacles

    run the demo with `map_size="large"`
    
## Simulation Results with Different Lower-level controllers:

More details about the controllers can be found in [this repositery](https://github.com/Mumamuye413/unicycle_controller_sim).

  * Goal position tracking controller [Cone] (forward motion only)
    
    Run the demo with `controller_type="Cone"` `bi-directional=False`

    <img src="/gif/sim3_8o_cone.gif" alt="8o_cone" width="600"/>

  * Goal position tracking controller [Cone] (bi-directional)

    Run the demo with `controller_type="Cone"` `bi-directional=True`
    
    <img src="/gif/sim4_8o_bdcone.gif" alt="8o_bdcone" width="600"/>

  * Goal pose tracking controller [Polar] (forward motion only)

    Run the demo with `controller_type="Polar"` `bi-directional=False`
    
    <img src="/gif/sim5_8o_polar.gif" alt="8o_polar" width="600"/>

  * Goal pose tracking controller [Polar] (bi-directional)

    Run the demo with `controller_type="Polar"` `bi-directional=True`
    
    <img src="/gif/sim6_8o_bdpolar.gif" alt="8o_bdpolar" width="600"/>



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
  
## Simulation Results with Different Environment Map
  * Small 1d map
    
    run the demo with `map_size="small"`
    
    [sim5_1d_cone.webm](https://github.com/Mumamuye413/actlpg_ref_gvn_sim/assets/97318853/0cb9d385-adf1-40e9-b25a-2c91bfa3e0a4)
    
  * Medium 2d map with fewer(two) moving obstacles

    run the demo with `map_size="medium"`

    [sim1_2o_dint.webm](https://github.com/Mumamuye413/actlpg_ref_gvn_sim/assets/97318853/95e75dbe-a611-47a1-ba60-c939371aad12)


  * Medium 2d map with more(eight) moving obstacles

    run the demo with `map_size="large"`
    
## Simulation Results with Different Lower-level controllers:

More details about the controllers can be found in [this repositery](https://github.com/Mumamuye413/unicycle_controller_sim).

  * Goal position tracking controller [Cone] (forward motion only)
    
    Run the demo with `controller_type="Cone"` `bi-directional=False`

    [sim3_8o_cone.webm](https://github.com/Mumamuye413/actlpg_ref_gvn_sim/assets/97318853/edf64ba7-72c3-422d-b921-fbb672a84428)

  * Goal position tracking controller [Cone] (bi-directional)

    Run the demo with `controller_type="Cone"` `bi-directional=True`
    
    [sim6_8o_bdcone.webm](https://github.com/Mumamuye413/actlpg_ref_gvn_sim/assets/97318853/ed4e9629-45a6-45d5-bca2-fb3e80e645bf)

  * Goal pose tracking controller [Polar] (forward motion only)

    Run the demo with `controller_type="Polar"` `bi-directional=False`
    
    [sim4_8o_polar.webm](https://github.com/Mumamuye413/actlpg_ref_gvn_sim/assets/97318853/7ae0e25b-1390-42fa-9c8e-05a2eeebf057)

  * Goal pose tracking controller [Polar] (bi-directional)

    Run the demo with `controller_type="Polar"` `bi-directional=True`
    
    [sim7_8o_bdpolar.webm](https://github.com/Mumamuye413/actlpg_ref_gvn_sim/assets/97318853/828cffe6-168c-495b-9ae2-39bb14c09969)



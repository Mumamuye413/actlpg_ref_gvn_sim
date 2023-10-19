# Safe Autonomous Robot Navigation in Dynamic Environments 
#### This repository contains code for a demonstration of safe robot navigation in a dynamic environment using Reference Governor and Control Barrier Function techniques. It is part of Zhuolin Niu's master thesis (https://www.proquest.com/openview/eb0055fcbc801cc62367fe1aaec13503/1?pq-origsite=gscholar&cbl=18750&diss=y).

#### In this simulation code, the differential drive robot dynamics was simulated with an unicycle model with control, and multiple moving obstacles are set to move back-and-forth with a constant speed along give trajectories. 

## Simulation Environment Map
  * Small 1d map
    
    run the demo with `map_size="small"`
    
  * Medium 2d map with fewer moving obstacles

    run the demo with `map_size="medium"`

  * Medium 2d map with more moving obstacles

    run the demo with `map_size="large"`
    
## Lower-level controllers:
  * Goal position tracking controller [Cone] (forward motion only)
    
    Run the demo with `controller_type="Cone"` `bi-directional=False`
    
  * Goal position tracking controller [Cone] (bi-directional)

    Run the demo with `controller_type="Cone"` `bi-directional=True`
    
  * Goal pose tracking controller [Polar] (forward motion only)

    Run the demo with `controller_type="Polar"` `bi-directional=False`
    
  * Goal pose tracking controller [Polar] (bi-directional)

    Run the demo with `controller_type="Polar"` `bi-directional=True`
    


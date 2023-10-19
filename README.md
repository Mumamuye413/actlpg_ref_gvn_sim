# Safe Autonomous Robot Navigation in Dynamic Environments 
#### This repository contains code for a demonstration of safe robot navigation in a dynamic environment using Reference Governor and Control Barrier Function techniques. It is part of Zhuolin Niu's [master thesis](https://escholarship.org/content/qt1jd778fm/qt1jd778fm.pdf).

#### In this simulation code, the differential drive robot dynamics was simulated with an unicycle model with control, and multiple moving obstacles are set to move back-and-forth with a constant speed along give trajectories. 

## Simulation Environment Map
  * Small 1d map
    
    run the demo with `map_size="small"`
    
    
    [![Video](https://i9.ytimg.com/vi_webp/pjxTLKdB4Ag/mq2.webp?sqp=CPSlxqkG&rs=AOn4CLAyxw9Et07H9ymKff5ykLZ_JacKVg)](https://youtu.be/pjxTLKdB4Ag)

  * Medium 2d map with fewer moving obstacles

    run the demo with `map_size="medium"`

    [![Video]
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
    


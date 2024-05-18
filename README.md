# Distribution-Generation-Placement-Study

This project aims to optimize the placement of distributed generators (DGs) in 
power systems. The method begins with the application of the forward-backward 
sweep algorithm on reference data. The output is then fed into the Combined 
Power Loss Sensitivity function, which selects the top 50% of buses most 
immune to power loss sensitivity. An objective function is then applied, which 
comprises three components: F1 for power loss minimization, F2 for total 
voltage deviation, and F3 for voltage stability. The final step involves the use of 
an improved Grey Wolf algorithm to determine the most suitable position for DG 
placement. This comprehensive approach ensures efficient and stable power 
distribution.

In conclusion, our study aimed to enhance the efficiency of locating Distributed 
Generators (DGs) within power systems by employing a hybrid optimization 
approach. Our methodology involved a twofold strategy, integrating both 
analytical and heuristic techniques. Beginning with data extraction from the 
IEEE sample dataset, we utilized the Forward-Backward Sweep method (FBS) 
and its variant, the CPLS-CPLSV, to focus our analysis on the top 50% of buses 
exhibiting significant FBS values.
Subsequently, we formulated an objective function and applied the Grey Wolf 
Optimization (GWO) algorithm, along with its improved version, to iteratively 
optimize the placement of DGs within the power network. Through multiple 
iterations, we identified the optimal P bus locations for two distinct systems: bus 7 
for the 33-bus system and bus 59 for the 69-bus system. Through multiple 
iterations, we identified the optimal PQ bus locations for two distinct systems: bus 7 
for the 33-bus system and bus 61 for the 69-bus system.
Our findings underscore the efficacy of the hybrid approach in pinpointing the 
most suitable buses for DG integration, thereby enhancing system reliability and 
minimizing power losses. 

# Carla

"The CARLA module is logically the first module and facilitates data extraction from the CARLA Simulator
. Using the [CARLA](https://pypi.org/project/carla/) Python package, it is possible to configure CARLA's 
simulation and export data such as semantic segmentation, depth information, and bounding boxes. To provide the 
pipeline user with complete flexibility for configuration, the module allows the loading of custom scripts called 
"Carla Scripts." These scripts are used to program the simulation. The interface "CarlaScriptInterface" is 
provided to help the user maintain the necessary structure.

The module provides the functions `pre()` and `post()`, which are called before and after the `run_script()` function 
to set up the basic environment and clean up. The `run_script()` method itself is called from the carla_module during 
the pipeline run, executing the script. 

Safety is a crucial topic, and it is important to acknowledge that this module should not be used in a public 
environment. The "Carla Scripts" are loaded without validation, which could allow malicious code to be executed 
during a pipeline run."

Please check out the original repository:
https://github.com/carla-simulator/carla

## Installation

Please follow the installation guide for the CARLA server and the CARLA python package found 
[here](https://github.com/carla-simulator/carla). After that write your own Carla Script and start the pipeline.
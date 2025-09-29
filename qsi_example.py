"""
Example
"""

import numpy as np
import uuid
from qsi.coordinator import Coordinator
from qsi.state import State, StateProp


# Initiate the Coordinator
coordinator = Coordinator()

# Start the module processes, before running the coordinator process
frequency_conversion = coordinator.register_component(
    module="qsi_module.py", runtime="python"
)

# Run the coordinator, when the coordinator starts
coordinator.run()

frequency_conversion.set_param("hbar", 1.0)
frequency_conversion.set_param("zi", 1.0)
frequency_conversion.set_param("kappa_a", 0.01)
frequency_conversion.set_param("kappa_b", 0.001)
frequency_conversion.set_param("tol", 1e-12)
frequency_conversion.set_param("coh", 0)
frequency_conversion.set_param("time_start", 0)
frequency_conversion.set_param("time_stop", 50)
frequency_conversion.set_param("time_steps", 5)

print(frequency_conversion.send_params())


"""
state_input = State(
    StateProp(state_type="light", truncation=3,
              wavelength=780, polarization="R")
)
state_output = State(
    StateProp(state_type="light", truncation=3,
              wavelength=1550, polarization="R")
)

sps_kraus, sps_kraus_spaces, error, retry = frequency_conversion.channel_query(
    state_input,
    {
        "input": state_input.state_props[0].uuid,
        "output": state_output.state_props[0].uuid,
    },
)
"""

coordinator.terminate()

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
    module="qsi_module_full_kraus.py", runtime="python"
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

frequency_conversion.send_params()

# --- Build single-mode states ---
s_in = State(
    StateProp(
        state_type="light",
        truncation=2,
        wavelength=780.0,
        polarization="R",
        bandwidth=1.0,
    )
)
s_in.state[:] = 0
s_in.state[1, 1] = 1.0  # |1><1|

s_out = State(
    StateProp(
        state_type="light",
        truncation=2,
        wavelength=1550.0,
        polarization="R",
        bandwidth=1.0,
    )
)
s_in.join(s_out)  # tensor order is [780, 1550]
composite = s_in

signals = [
    {"role": "input", "uuid": composite.state_props[0].uuid},
    {"role": "output", "uuid": composite.state_props[1].uuid},
]

response, operators = frequency_conversion.channel_query(
    composite,
    port_assign=None,
    signals=signals,
)

print("==============================================================")
print(f"Created {len(operators)}")
print("INFO")
print(response["info"])

print("Applying Kraus")
kraus_spaces = composite.get_all_props(response["kraus_state_indices"])
# By contract these are [input_uuid, output_uuid]
composite.apply_kraus_operators(operators, kraus_spaces)
out_uuid = response["kraus_state_indices"][1]
out_prop = composite.get_props(out_uuid)
rho_out = composite.get_reduced_state([out_prop])
print("Output mode density matrix:\n", rho_out)


coordinator.terminate()

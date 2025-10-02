"""
Example
"""

import numpy as np
import uuid
from qsi.coordinator import Coordinator
from qsi.state import State, StateProp


# Initiate the Coordinator
coordinator = Coordinator(8474)

# Start the module processes, before running the coordinator process
frequency_conversion = coordinator.register_component(
    module="qsi_module_traced_kraus.py", runtime="python"
)

# Run the coordinator, when the coordinator starts
coordinator.run()

frequency_conversion.set_param("hbar", 1.0)
frequency_conversion.set_param("zi", 1.0)
frequency_conversion.set_param("kappa_a", 0.01)
frequency_conversion.set_param("kappa_b", 0.001)
frequency_conversion.set_param("g_sc", 0.8)
frequency_conversion.set_param("g_Raman_s", 1e-5)
frequency_conversion.set_param("g_Raman_c", 1e-4)
frequency_conversion.set_param("tol", 1e-12)
frequency_conversion.set_param("coh", 0)
frequency_conversion.set_param("time_start", 0)
frequency_conversion.set_param("time_stop", 50)
frequency_conversion.set_param("time_steps", 5)


frequency_conversion.send_params()

# --- Build single-mode states ---
signal = State(
    StateProp(
        state_type="light",
        truncation=2,
        wavelength=780.0,
        polarization="R",
        bandwidth=1.0,
    )
)
signal.state[:] = 0
signal.state[1, 1] = 1.0  # |1><1|

signals = [
    {
        "role": "input",
        "uuid": signal.state_props[0].uuid,  # 780 nm
    }  # 1550 nm
]

response, operators = frequency_conversion.channel_query(
    signal,
    port_assign=None,
    signals=signals,
)

print("==============================================================")
print(f"Created {len(operators)}")
print("INFO")
print(response["info"])

print("Applying Kraus")
kraus_spaces = signal.get_all_props(response["kraus_state_indices"])

# By contract these are [input_uuid]
signal.apply_kraus_operators(operators, kraus_spaces)
out_uuid = response["kraus_state_indices"][0]

prop = signal.get_props(out_uuid)
prop.wavelength = 1550

out_prop = signal.get_props(out_uuid)
rho_out = signal.get_reduced_state([out_prop])
print("Output mode density matrix:\n", rho_out)


coordinator.terminate()

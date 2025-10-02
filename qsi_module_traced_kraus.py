"""
QSI implementation for the frequency conversion module
"""

from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict
import traceback

from qsi.qsi import QSI
from qsi.helpers import numpy_to_json
from qsi.state import State, StateProp
import time
import numpy as np

from channel_model import Kraus_DFG

# Initiate the QSI object instance
qsi = QSI()


@dataclass()
class Parameters:
    hbar: float
    zi: float
    gamma_s: float
    gamma_c: float
    g_sc: float
    g_Raman_s: float
    g_Raman_c: float
    tol: float
    coh: float
    time_start: float
    time_stop: float
    time_steps: float
    Ns: int = 7
    Nc: int = 6
    signal_photon: int = 1
    convsn_photon: int = 0

    def to_dict(self) -> dict:
        """Transform into dict format expected by Kraus_DFG."""
        d = asdict(self)
        # Replace time_start/stop/steps with actual time vector
        d["time"] = np.linspace(self.time_start, self.time_stop, int(self.time_steps))
        # Remove unused keys if needed
        d.pop("time_start", None)
        d.pop("time_stop", None)
        d.pop("time_steps", None)
        return d


# Here we store the parameters
PARAMETERS: Optional[Parameters] = None

# HELPERS


def _extract_param_values(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinator sends:
      "params": {"name": {"value"}, "type": "..."}
    This method converts that to a flat dict of Python scalars
    """
    out = {}
    for k, v in raw.items():
        if not isinstance(v, dict) or "value" not in v:
            out[k] = v
            continue
        val = v["value"]
        if (
            isinstance(val, (list, tuple))
            and len(val) == 2
            and all(isinstance(x, (int, float)) for x in val)
        ):
            out[k] = complex(val[0], val[1])
        else:
            out[k] = val
    return out


def get_io_from_signals(signals):
    in_uuid = out_uuid = None
    for s in signals or []:
        if isinstance(s, dict):
            # role + uuid items
            if s.get("role") == "input" and "uuid" in s:
                in_uuid = s["uuid"]
            elif s.get("role") == "output" and "uuid" in s:
                out_uuid = s["uuid"]
            # flat input/output uuids
            if "input_uuid" in s and "output_uuid" in s:
                in_uuid, out_uuid = s["input_uuid"], s["output_uuid"]
            # nested
            if "io" in s and isinstance(s["io"], dict):
                io = s["io"]
                if "input_uuid" in io and "output_uuid" in io:
                    in_uuid, out_uuid = io["input_uuid"], io["output_uuid"]
    return in_uuid, out_uuid


def info_minimal(info: dict) -> dict:
    return {
        "operator_num": int(info.get("operator_num_M", 0)),
        "dimensions": int(info.get("dimensions_M", 0)),
        "norm_error": float(info.get("norm_error", 0.0)),
        "runtime": float(info.get("runtime", 0.0)),
    }


@qsi.on_message("state_init")
def state_init(msg):
    """
    This module declares no internal state
    """
    return {"msg_type": "state_init_response", "states": [], "state_ids": []}


@qsi.on_message("param_query")
def param_query(msg):
    return {
        "msg_type": "param_query_response",
        "params": {
            "Ns": "number",
            "Nc": "number",
            "hbar": "number",
            "zi": "number",
            "gamma_s": "number",
            "gamma_c": "number",
            "g_sc": "number",
            "g_Raman_s": "number",
            "g_Raman_c": "number",
            "tol": "number",
            "coh": "number",
            "time_start": "number",
            "time_stop": "number",
            "time_steps": "number",
        },
    }


@qsi.on_message("param_set")
def param_set(msg):
    global PARAMETERS
    try:
        flat = _extract_param_values(msg["params"])
        PARAMETERS = Parameters(**flat)
    except Exception as e:
        return {"msg_type": "param_set_response", "message": str(e)}
    return {
        "msg_type": "param_set_response",
    }


@qsi.on_message("channel_query")
def channel_query(msg):
    global PARAMETERS
    assert isinstance(PARAMETERS, Parameters)

    # 1. Get the input/output states from the message
    # 2. Verify that the input/output props match (input/output frequency wl)
    # 3. Update the PARAMETERS with the given truncation
    # 4. Verify that enough dimensions are present in the state

    # Rebuild a full state from the message payload
    in_state = State.from_message(msg)
    signals = msg.get("signals", [])
    in_uuid, out_uuid = get_io_from_signals(signals)
    if not (in_uuid):
        print(f"{in_uuid}, {out_uuid}", flush=True)
        return {
            "msg_type": "channel_query_response",
            "message": "Provide input UUIDs in 'signals' (e.g."
            "[{'role': 'input', 'uuid': ...}]",
        }

    in_prop = in_state.get_props(in_uuid)

    # We create a synthetic state to compute the Kraus

    synth_state_prop = StateProp(
        state_type="light",
        truncation=in_prop.truncation,
        wavelength=1550,
        polarization=in_prop.polarization,
        bandwidth=in_prop.wavelength,
    )
    synth_state = State(synth_state_prop)

    # Tensor product the synth state into the given state
    in_state.join(synth_state)
    full_state = in_state

    full_state._reorder([in_prop, synth_state_prop])

    initial_state = full_state.get_reduced_state([in_prop, synth_state_prop])
    # Construct the initial state by tracing out the states that do not matter
    # Reshape states so that input is first and output second in the product

    # Update the trunctaion in the parameters
    PARAMETERS.Ns = int(in_prop.truncation)
    PARAMETERS.Nc = int(synth_state_prop.truncation)
    try:
        kraus_list, m_list, info, fidelity, *_ = Kraus_DFG(
            params=PARAMETERS.to_dict(), initial_state=initial_state
        )
        return {
            "msg_type": "channel_query_response",
            "kraus_operators": [numpy_to_json(M) for M in m_list],
            "Fidelity": fidelity,
            "kraus_state_indices": [in_uuid],
            "error": float(info.get("norm_error", 0.0)),
            "retrigger": False,
            "retrigger_time": 0,
            "info": info_minimal(info),
        }
    except Exception as e:
        print("=== Exception in channel_query ===")
        traceback.print_exc()
        print("=================================")
        return {
            "msg_type": "channel_query_response",
            "simulation_error": "Something went wrong in the simulation process",
            "message": str(e),
        }


@qsi.on_message("terminate")
def terminate(msg):
    qsi.terminate()


qsi.run()
time.sleep(1)

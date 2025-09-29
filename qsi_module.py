"""
QSI implementation for the frequency conversion module
"""

from dataclasses import dataclass
from typing import Optional

from qsi.qsi import QSI
from qsi.helpers import numpy_to_json, pretty_print_dict
from qsi.state import State, StateProp
import time
import numpy as np
import uuid
import scipy.linalg as la
from scipy.special import factorial

from channel_model import Kraus_DFG

# Initiate the QSI object instance
qsi = QSI()


@dataclass()
class Parameters:
    hbar: float
    zi: float
    kappa_a: float
    kappa_b: float
    tol: float
    coh: float
    time_start: float
    time_end: float
    time_steps: float
    Ns: int = 7
    Nc: int = 6


# Here we store the parameters
PARAMETERS: Optional[Parameters] = None


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
            "kappa_a": "number",
            "kappa_b": "number",
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
    params = msg["params"]
    try:
        PARAMETERS = Parameters(**params)
        print(PARAMETERS)
    except Exception as e:
        print("EXCEPTION")
        return {"msg_type": "param_set_response", "message": str(e)}
    return {
        "msg_type": "param_set_response",
    }


@qsi.on_message("channel_query")
def channel_query(msg):
    global PARAMETERS
    assert isinstance(PARAMETERS, Parameters)
    state = State.from_message(msg)

    # 1. Get the input/output states from the message
    # 2. Verify that the input/output props match (input/output frequency wl)
    # 3. Update the PARAMETERS with the given truncation
    # 4. Verify that enough dimensions are present in the state

    # Construct the initial state by tracing out the states that do not matter
    # Reshape states so that input is first and output second in the product
    initial_state = np.array([])

    # Update the trunctaion in the parameters
    PARAMETERS.Nc = 0
    PARAMETERS.Ns = 0

    try:
        kraus, info, _, _, _ = Kraus_DFG(
            params=PARAMETERS, initial_state=initial_state)

        return {
            "msg_type": "channel_query_response",
            "kraus_operators": [numpy_to_json(x) for x in kraus],
            "kraus_state_indices": [uuid],
            "error": info["trace_error"],
            "retrigger": False,
            "info": info,
        }
    except Exception as e:
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

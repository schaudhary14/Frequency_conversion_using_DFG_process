# FREQUENCY CONVERSION
The frequency conversion device is based on the $\chi^{(2)}$ nonlinear process. It emulates the Difference frequency generation (DFG) process, where the pump at frequency $\omega_p$
and an input signal state at frequency $\omega_s$ is converted to a new output state at frequency $\omega_c = \omega_p - \omega_s$.

The input state is a bipartite system of signal and converted mode | \psi \rangle = | signal \rangle |0 \rangle


## How to use
In order to run the examples you can use either the full Kraus channel implementation or the reduced implementation
### Install dependencies
```bash
pip install -r reguirements.txt
```
### To use the full implementation run
```bash
python qsi_example_full_kraus.py 1234
```
Select any unoccupied port, here `1234` example is used.

### To run the reduced implementation run
```bash
python qsi_example_traced_kraus.py 1234
```


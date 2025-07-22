

### Variational optimization of iPEPS and iPESS wavefunctions with the C3 honeycomb CTMRG

Computes the energy of the Heisenberg model on the honeycomb and triangular lattice using iPEPS and iPESS wavefunctions defined on a honeycomb lattice. The optimisation is made by minimising the energy with pytorch. The energy is computed by contracting the 2D tensor network with the C3 symmetric honeycomb corner transfer matrix renormalisation group algorithm (https://arxiv.org/abs/2306.09046).

This implementation is built upon https://github.com/wangleiphy/tensorgrad/blob/master/. The main difference is the computation of the energy using the honeycomb CTMRG rather then the square one.
 
Installation:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

To use the program, run the following command:
```bash
python main.py -D=3 -chi=30 -dtype="float64" --model="PESS"
``` 
for the triangular lattice and 
```bash
python main.py -D=3 -chi=30 -dtype="float64" --model="PEPS"
``` 
for the honeycomb lattice.
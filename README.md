# RamsesNN

**Embedding neural networks into dynamic power system simulators**

RamsesNN integrates Physics-Informed Neural Networks (PINNs) into the RAMSES time-domain simulator, part of the [STEPSS](https://stepss.sps-lab.org/) power system simulation platform. Neural networks trained in PyTorch are exported to ONNX, converted to native Fortran with the [roseNNa](https://github.com/comp-physics/roseNNa) inference library, and embedded in RAMSES as custom injector models — replacing or augmenting traditional power system component models.

---

## Features

- **PyTorch-to-Fortran pipeline** — train in PyTorch, export to ONNX, run natively inside the Fortran simulator via roseNNa
- **Fast inference** — roseNNa's optimized Fortran implementation is 2–5x faster than PyTorch for the small networks typical of physics applications
- **Minimal intrusion** — the network plugs into RAMSES as a standard user injector model (Norton equivalent), with no changes to the simulator core
- **Full and reduced-order models** — example synchronous machine PINNs with 10-input/9-output (full) and 8-input/7-output (reduced) variants
- **Multiple named models** — the roseNNa reader shipped here is modified so `initialize_nnx(model_name)` loads `onnxModel_<name>.txt` / `onnxWeights_<name>.txt`, allowing several networks side by side
- **Standalone test harness** — `Evaluate_PINN/` evaluates the PINN + roseNNa inference outside the simulator
- **ONNX standard** — universal format compatible with PyTorch, TensorFlow, and Keras

---

## Installation

**Requirements:** Python 3.10 with `numpy<2`, `onnx`, `onnxruntime`, CPU `torch`, and `fypp` (for ONNX-to-Fortran conversion); Visual Studio with Intel Fortran (oneAPI) to build the included `URAMSES` and `Evaluate_PINN` solutions on Windows; the RAMSES library and module files from [stepss-uramses](https://github.com/SPS-L/stepss-uramses) (proprietary, not included — see [License](#license)).

```bash
git clone https://github.com/SPS-L/stepss-RamsesNN.git
```

Create the Python environment used by the conversion pipeline:

```bash
conda create -n roparse python=3.10 -y
conda activate roparse

pip install "numpy<2" onnx==1.15.0 onnxruntime==1.16.3
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fypp
```

A full copy of roseNNa is vendored at `Evaluate_PINN/roseNNa-master/`; you can use it directly or a separate [roseNNa](https://github.com/comp-physics/roseNNa) checkout.

---

## Quick Start

1. **Train and export** your network in PyTorch to ONNX (`torch.onnx.export`)
2. **Convert** the ONNX model to Fortran with roseNNa (`modelParserONNX.py` + `fypp`) — see the [workflow](#complete-workflow) below
3. **Drop the generated files** into the RAMSES project: `modelCreator.f90` alongside the roseNNa sources in `URAMSES/rosenna/`, and `onnxModel_<name>.txt` / `onnxWeights_<name>.txt` next to the RAMSES executable
4. **Build** the `URAMSES` solution and run the simulation; the `inj_norton` injector model performs the neural network forward pass at every step

A ready-made reduced-order machine model is included: `URAMSES/rosenna/` ships `modelCreator.f90`, `onnxModel_redv2.txt`, `onnxWeights_redv2.txt`, and the source `PINN_redv2.onnx`. The full-model files (`full_frzd`) must be generated from your own trained network.

---

## Complete Workflow

### 1. Train and Export Neural Network

Train your neural network in PyTorch and export it to ONNX format:

```python
import torch

# Train your model
model = YourNeuralNetwork()
# ... training code ...

# Export to ONNX
init_cond = torch.randn(1, 10)  # Example input matching your network's input shape
torch.onnx.export(model, init_cond, "NN.onnx")
```

### 2. Convert ONNX Model with roseNNa

Work in roseNNa's `fLibrary/` directory (e.g. `Evaluate_PINN/roseNNa-master/fLibrary/`), with the `roparse` environment active.

#### Shape Inference

ONNX models need shape inference for proper parsing. Create and run a helper script:

**Bash (Linux/Mac):**
```bash
cat > infer_shapes.py << 'EOF'
import onnx, onnx.shape_inference
m = onnx.load('./NN.onnx')
m_inf = onnx.shape_inference.infer_shapes(m)
onnx.save(m_inf, './NN_inferred.onnx')
print('Wrote NN_inferred.onnx')
EOF
python infer_shapes.py
```

**PowerShell (Windows):**
```powershell
$code = @"
import onnx, onnx.shape_inference
m = onnx.load(r'.\NN.onnx')
m_inf = onnx.shape_inference.infer_shapes(m)
onnx.save(m_inf, r'.\NN_inferred.onnx')
print('Wrote NN_inferred.onnx')
"@
Set-Content -Encoding ASCII infer_shapes.py $code
python .\infer_shapes.py
```

#### Parse and Generate Fortran Code

```bash
# Clean previous files
rm -f onnxModel.txt onnxWeights.txt variables.fpp modelCreator.f90

# Parse ONNX model
python modelParserONNX.py -f "./NN.onnx" -w "./NN.onnx" -i "./NN_inferred.onnx"

# Verify output files were created
ls onnxModel.txt onnxWeights.txt variables.fpp

# Generate Fortran code via fypp preprocessor
python -m fypp ./modelCreator.fpp ./modelCreator.f90

# Fix C++ style namespace syntax for Fortran
sed -i 's/onnx:://g' modelCreator.f90  # Linux/Mac
# OR for Windows PowerShell:
# (Get-Content .\modelCreator.f90) -replace 'onnx::','' | Set-Content .\modelCreator.f90
```

#### Compile roseNNa Library

```bash
make library
```

This generates `libcorelib.a`, the core roseNNa library file.

---

## Integration into RAMSES

### Required roseNNa Files

These roseNNa source files must be part of the RAMSES user-model project. They are already included in `URAMSES/rosenna/`:

- `activation_funcs.f90`
- `derived_types.f90`
- `layers.f90`
- `reader.f90`
- `rosenna.f90`
- `modelCreator.f90` (generated for your specific model)
- `aux_functions.f90` (RamsesNN helper routines: Park transforms, machine solver, infinite-bus equations)

### Required Data Files

Place the generated model files in the RAMSES executable directory:

- `onnxModel_<name>.txt` — model structure
- `onnxWeights_<name>.txt` — model weights

**Note**: The `reader.f90` in this repository is modified relative to upstream roseNNa: `initialize_nnx(model_name)` takes a model name and loads the matching `onnxModel_<name>.txt` / `onnxWeights_<name>.txt` pair, so multiple networks can coexist.

### Example: Norton Injector Model

`URAMSES/my_models/inj_norton.f90` demonstrates a complete integration. Key components:

#### Module Setup

```fortran
module inj_norton_mod
    use iso_fortran_env, only: real64, int32
    use MODELING
    use rosenna      ! Neural network library
    implicit none

    ! Neural network I/O arrays
    real(real64), allocatable :: pinn_input(:,:)    ! (1, n_in)
    real(real64), allocatable :: pinn_output(:,:)   ! (1, n_out)
```

#### Initialization

```fortran
case (initialize)
    ! Define model dimensions (full or reduced)
    if (prm(5) < 1.5) then
        n_nn_input = 10
        n_nn_output = 9
    else
        n_nn_input = 8
        n_nn_output = 7
    end if

    ! Allocate arrays
    allocate(pinn_input(1, n_nn_input))
    allocate(pinn_output(1, n_nn_output))

    ! Initialize input with steady-state values
    pinn_input(1,1) = 0.01_real64      ! timestep
    ! ... additional states ...

    ! Initialize roseNNa with model name
    nn_model_name = "full_frzd"  ! or "redv2" for reduced model
    call initialize_nnx(nn_model_name)
```

#### Inference Execution

```fortran
case (evaluate_eqs)
    ! Update inputs based on current system state
    call inf_bus_equations(vx, vy, old_ix, old_iy, Xline, Rline, &
                          new_v_infty, new_v_infty_ang)

    pinn_input(1,8) = new_v_infty
    pinn_input(1,9) = new_v_infty_ang

    ! Neural network forward pass
    call use_model(pinn_input, pinn_output)

    ! Convert NN output to physical quantities
    call machine_solver(pinn_output(1,1:9), machine_output(1,1:6))

    ! Transform to xy coordinates
    call park_transform_dq_xy(machine_output(1,3), machine_output(1,4), &
                             pinn_output(1,1), ixnorton, iynorton)
```

---

## Platform-Specific Instructions

### Windows with Visual Studio 2022

1. **Add source files** to your project:
   - Right-click project → Add → Existing Item
   - Select the roseNNa `.f90` files listed above

2. **Configure module path**:
   - Project → Properties → Fortran → General → Additional Include Directories
   - Add path to roseNNa's `objFiles/` directory

3. **Link library**:
   - Project → Properties → Linker → Input → Additional Dependencies
   - Add path to `libcorelib.a`

See `URAMSES/README.rst` for the general procedure to build user models against RAMSES with Visual Studio and Intel Fortran.

---

## Repository Structure

```
RamsesNN/
├── Evaluate_PINN/                 # Standalone Fortran console app: test PINN inference outside RAMSES
│   └── roseNNa-master/            # Vendored copy of the roseNNa library (MIT)
├── URAMSES/                       # RAMSES user-model solution (Visual Studio / Intel Fortran)
│   ├── src/                       # Model registration files (usr_inj_models.f90, ...)
│   ├── my_models/                 # User models, incl. inj_norton.f90 (NN-driven Norton injector)
│   └── rosenna/                   # roseNNa sources + generated model code and weights (redv2)
├── LICENSE
└── README.md
```

---

## Model Types Supported

### Full Model (10 inputs, 9 outputs)
- Inputs: timestep, δ, ω, e_d, e_q, ψ_d, ψ_q, V_∞, θ_∞, v_f
- Outputs: δ, ω, e_d, e_q, ψ_d, ψ_q, V_∞, θ_∞, v_f

### Reduced Model (8 inputs, 7 outputs)
- Inputs: timestep, δ, ω, e_d, e_q, V_∞, θ_∞, E_f
- Outputs: δ, ω, e_d, e_q, V_∞, θ_∞, E_f

---

## Technical Details

### Neural Network Architecture Support

roseNNa currently supports:
- Multi-Layer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs/LSTMs)

### Performance

For small networks typical in physics applications:
- **2–5x faster** than PyTorch inference
- Optimized for Fortran/C HPC environments
- Minimal memory footprint

### LSTM/RNN Conversion Note

When converting LSTMs to ONNX, you need two exports (with and without constant folding):

```python
# Model structure (with optimization)
torch.onnx.export(model, (inp, hidden),
                  "model_structure.onnx",
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input', 'hidden_state', 'cell_state'],
                  output_names=['output'])

# Model weights (without optimization)
torch.onnx.export(model, (inp, hidden),
                  "model_weights.onnx",
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=False,
                  input_names=['input', 'hidden_state', 'cell_state'],
                  output_names=['output'])
```

---

## References

- B. Gelfort (2025). "Integrating neural networks into dynamic power system simulators", MSc Thesis, ETH Zurich.
- A. Bati, S. H. Bryngelson (2024). "RoseNNa: A performant, portable library for neural network inference with application to computational fluid dynamics". *Computer Physics Communications*. [DOI: 10.1016/j.cpc.2023.109052](https://doi.org/10.1016/j.cpc.2023.109052)
- P. Aristidou, S. Lebeau, and T. Van Cutsem (2016). "Power system dynamic simulations using a parallel two-level schur-complement decomposition". *IEEE Transactions on Power Systems*. [doi: 10.1109/TPWRS.2015.2509023](https://doi.org/10.1109/TPWRS.2015.2509023)

---

## Documentation

| Document | Description |
|----------|-------------|
| [STEPSS platform](https://stepss.sps-lab.org/) | Documentation site for the STEPSS simulation platform |
| [PyRAMSES](https://stepss.sps-lab.org/pyramses/) | Using the compiled RAMSES library from Python |
| [URAMSES/README.rst](URAMSES/README.rst) | Building user models with Visual Studio and Intel Fortran |
| [roseNNa](https://github.com/comp-physics/roseNNa) | Upstream neural network inference library for Fortran/C |

---

## License

RamsesNN is distributed under the **MIT License** — see [LICENSE](LICENSE). Copyright (c) 2025 Bruno Gelfort.

The MIT grant covers **only** the code in this repository. It does not extend to
its dependencies, which carry their own terms:

- **roseNNa** — MIT License
- **RAMSES / URAMSES** — Academic Public License (non-commercial). RAMSES is the
  property of the University of Liège.

### Obtaining RAMSES

The compiled RAMSES library (`libramses.lib` / `libramses.a`) and its Fortran
module files are **not** included in this repository: they are proprietary and
not redistributable under MIT. To build the URAMSES models here, get them from
[stepss-uramses](https://github.com/SPS-L/stepss-uramses) and place them in
`URAMSES/modules/` (Windows/Intel) or `URAMSES/modules_lin/` (Linux/gfortran).

---

## Authors

RamsesNN was originally developed by **Bruno Gelfort** (MSc thesis, ETH Zurich, 2025). It is maintained by the [Sustainable Power Systems Laboratory (SPS-L)](https://sps-lab.org/) at the Cyprus University of Technology, under the direction of Dr. Petros Aristidou.

**Contact:** info@sps-lab.org — [https://sps-lab.org](https://sps-lab.org)

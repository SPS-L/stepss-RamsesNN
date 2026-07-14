# RamsesNN

**Embedding Neural Networks into Dynamic Power System Simulators**

This repository contains the implementation and workflow for integrating Physics-Informed Neural Networks (PINNs) into the RAMSES power system simulator using the [roseNNa](https://github.com/comp-physics/roseNNa) inference library.

## 📋 Overview

RamsesNN enables the use of neural networks trained in PyTorch to replace or augment traditional power system component models in RAMSES simulations. The workflow converts PyTorch models to ONNX format, processes them with roseNNa, and integrates them into RAMSES through custom Fortran injector models.

### Key Features

- **Fast Inference**: Uses roseNNa's optimized Fortran implementation (2-5x faster than PyTorch for small networks)
- **Minimal Intrusion**: Seamlessly integrates into existing RAMSES workflows
- **Flexible Models**: Supports both full and reduced-order power system models
- **ONNX Standard**: Universal format compatible with PyTorch, TensorFlow, and Keras

### Related Projects

- [roseNNa](https://github.com/comp-physics/roseNNa) - Fast neural network inference library for Fortran/C
- [URAMSES](https://github.com/SPS-L/stepss-uramses) - RAMSES power system simulator distribution

---

## 🚀 Complete Workflow

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

Navigate to the `fLibrary/` directory in your roseNNa installation and follow these steps:

#### Prerequisites

Install required dependencies:

```bash
# Create conda environment for ONNX parsing
conda create -n roparse python=3.10 -y
conda activate roparse

# Install dependencies
pip install "numpy<2" onnx==1.15.0 onnxruntime==1.16.3
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fypp
```

#### Shape Inference

ONNX models need shape inference for proper parsing. Create and run a helper script:

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

## 🔧 Integration into RAMSES

### Required roseNNa Files

Add these files from roseNNa's `fLibrary/` directory to your RAMSES project (URAMSES):

- `activation_funcs.f90`
- `derived_types.f90`
- `layers.f90`
- `reader.f90`
- `rosenna.f90`
- `modelCreator.f90` (generated for your specific model)

### Required Data Files

Place these generated files in your RAMSES executable directory:

- `onnxModel.txt` - Model structure
- `onnxWeights.txt` - Model weights

**Note**: If using multiple neural networks simultaneously, you'll need to modify the reader logic in `rosenna.f90` to handle multiple model files.

### Example: Norton Injector Model

The `inj_norton.f90` model demonstrates a complete integration. Key components:

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
    pinn_input(1,2) = -0.81854824_real64  ! delta
    pinn_input(1,3) = omega            ! omega
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

## 🖥️ Platform-Specific Instructions

### Windows with Visual Studio 2022

1. **Add source files** to your project:
   - Right-click project → Add → Existing Item
   - Select the 6 roseNNa `.f90` files listed above

2. **Configure module path**:
   - Project → Properties → Fortran → General → Additional Include Directories
   - Add path to roseNNa's `objFiles/` directory

3. **Link library**:
   - Project → Properties → Linker → Input → Additional Dependencies
   - Add path to `libcorelib.a`

---

## 📁 Repository Structure

```
RamsesNN/
├── README.md                      # This file
├── URAMSES/                       # RAMSES simulator files
│   └── my_models/
│       └── inj_norton.f90        # Example NN-integrated injector model
└── [additional model files]
```

---

## 🔬 Model Types Supported

### Full Model (10 inputs, 9 outputs)
- Inputs: timestep, δ, ω, e_d, e_q, ψ_d, ψ_q, V_∞, θ_∞, v_f
- Outputs: δ, ω, e_d, e_q, ψ_d, ψ_q, V_∞, θ_∞, v_f

### Reduced Model (8 inputs, 7 outputs)
- Inputs: timestep, δ, ω, e_d, e_q, V_∞, θ_∞, E_f
- Outputs: δ, ω, e_d, e_q, V_∞, θ_∞, E_f

---

## 📚 Technical Details

### Neural Network Architecture Support

roseNNa currently supports:
- Multi-Layer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs/LSTMs)

### Performance

For small networks typical in physics applications:
- **2-5x faster** than PyTorch inference
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

## 📖 References

- B. Gelford (2025). "Integrating neural networks into dynamic power system simulators", MSc Thesis, ETH Zurich.
- A. Bati, S. H. Bryngelson (2024). "RoseNNa: A performant, portable library for neural network inference with application to computational fluid dynamics". *Computer Physics Communications*. [DOI: 10.1016/j.cpc.2023.109052](https://doi.org/10.1016/j.cpc.2023.109052)
- P. Aristidou, S. Lebeau, and T. Van Cutsem (2016). "Power system dynamic simulations using a parallel two-level schur-complement decomposition". *IEEE Transactions on Power Systems*. [doi: 10.1109/TPWRS.2015.2509023](https://doi.org/10.1109/TPWRS.2015.2509023)

---

## 📧 Contact

**Author:** Bruno Gelfort
**Web:** [https://sps-lab.org](https://sps-lab.org)  
**Email:** info@sps-lab.org

---

## 📄 License

The code written for this project is released under the MIT License — see [LICENSE](LICENSE).

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

## 🚧 Future Work

Planned additions to this repository:
- Complete PINN training pipeline documentation
- Additional example models for various power system components
- Automated testing framework
- Performance benchmarking tools
- Multi-model integration examples

<div align=center><img src="assets/vb-mitigator logo_250.png" width="20%" ><div align=left>

# Visual Bias Mitigator (VB-Mitigator)


The Visual Bias Mitigator is an open-source framework designed to empower researchers in the field of bias mitigation in  computer vision. This codebase provides a comprehensive environment where users can easily implement, run, and evaluate existing visual bias mitigation methods.

With the increasing awareness of bias in AI systems, it is crucial for researchers to have access to robust tools that facilitate the exploration and development of mitigation approaches. The Visual Bias Mitigator (VB-Mitigator) serves this purpose by offering:

- 🚀 **Implemented Methods**: A collection of established visual bias mitigation methods that can be directly utilized, allowing researchers to replicate and understand their functionality.
- 🔧 **Extensibility**: Researchers can exploit this code-base to develop custom bias mitigation approaches tailored to their specific needs. The framework is designed with flexibility in mind, enabling easy integration of new approaches.
- 📊 **Performance Comparison**: The framework facilitates the performance comparison between custom methods and state-of-the-art. 

The aim of this repository is to facilitate research in the domain of visual bias mitigation. By providing a comprehensive codebase that allows researchers to easily implement and build upon existing methodologies, we encourage the development of new approaches for addressing biases in computer vision tasks.

## Quickstart

Get started with Visual Bias Mitigator quickly:

### 1. Clone the Git Repository

```bash
git clone https://github.com/gsarridis/vb-mitigator.git
```

### 2. Create a Virtual Environment and Install Required Packages

You can use either `pip` or `conda` to create a virtual environment and install dependencies:

```bash
# create a virtual conda environment
conda create -n vb-mitigator python=3.11

# activate the environment
conda activate vb-mitigator

# install the required packages
pip install -r requirements.txt
```

### 3. Run a Sample Script

```bash
# run BAdd method on UTKFace dataset
bash ./scripts/utkface/badd/badd.sh
```

### 4. Check Logs for Results and Metrics  

The output is stored in the `outputs/utkface_baselines/badd` directory.

#### **Output Structure:**

```
├── outputs
│   ├── utkface_baselines
│   │   ├── badd
│   │   │   ├── logs.csv
│   │   │   ├── out.log
│   │   │   ├── best.pth
│   │   │   ├── latest.pth
│   │   │   └── train.events
```

## Documentation
You can find the complete documentation for VB-Mitigator [here](https://vb-mitigator.readthedocs.io/).
## Citations

```
@article{sarridis2024flac,
  title={Flac: Fairness-aware representation learning by suppressing attribute-class associations},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}

@article{sarridis2024badd,
  title={BAdd: Bias Mitigation through Bias Addition},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  journal={arXiv preprint arXiv:2408.11439},
  year={2024}
}

@article{sarridis2024mavias,
  title={MAVias: Mitigate any Visual Bias},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  journal={arXiv preprint arXiv:2412.06632},
  year={2024}
}
```

**Maintainer:** Ioannis Sarridis (gsarridis@iti.gr)<br>

## Acknowledgments
This research was supported by the EU Horizon Europe projects MAMMOth
(Grant Agreement 101070285) and ELIAS (Grant Agreement 101120237).
<div align="center"> <img src="assets/mammoth_logo.svg" width="20%" alt="MAMMOth Project Logo"> <img src="assets/elias_logo.png" width="20%" alt="ELIAS Project Logo"> </div>
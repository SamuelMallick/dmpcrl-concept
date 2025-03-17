# Multi-Agent Reinforcement Learning via Distributed MPC as a Function Approximator

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/dmpcrl-concept/blob/main/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Multi-Agent Reinforcement Learning via Distributed MPC as a Function Approximator](https://www.sciencedirect.com/science/article/pii/S0005109824002978) published in [Automatica](https://www.sciencedirect.com/journal/automatica/vol/160/suppl/C).

In this work we propose the use of a distributed model predictive control scheme as a function approximator for multi-agent reinforcement learning. We consider networks of linear dynamical systems.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{mallick2024multi,
  title={Multi-agent reinforcement learning via distributed MPC as a function approximator},
  author={Mallick, Samuel and Airaldi, Filippo and Dabiri, Azita and De Schutter, Bart},
  journal={Automatica},
  volume={167},
  pages={111803},
  year={2024},
  publisher={Elsevier}
}
```

---

## Installation

The code was created with `Python 3.11`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/dmpcrl-concept.git
cd dmpcrl-concept
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`data`** contains the .pkl data files that have been generated for the paper Multi-Agent Reinforcement Learning via Distributed MPC as a Function Approximator.
- **`plotting`** contains scripts that are used for generating the images used in the paper Multi-Agent Reinforcement Learning via Distributed MPC as a Function Approximator.
- **`power_system`** contains contains all files relating to the power system example in the paper Multi-Agent Reinforcement Learning via Distributed MPC as a Function Approximator. q_learning_power.py runs the MARL training algorithm.
- **`example_1`** contains contains all files relating to the power system example in the paper Multi-Agent Reinforcement Learning via Distributed MPC as a Function Approximator. example_1.py.py runs the MARL training algorithm.
## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/dmpcrl-concept/blob/main/LICENSE) file included with this repository.

---

## Author

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2023 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “dmpcrl-concept” (Multi-Agent Reinforcement Learning via Distributed MPC as a Function Approximator) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.

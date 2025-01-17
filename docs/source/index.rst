.. toctree::
   :maxdepth: 2
   :caption: DOCUMENTATION

Overview
========

The Visual Bias Mitigator is an open-source framework designed to empower researchers in the field of bias mitigation in  computer vision. This codebase provides a comprehensive environment where users can easily implement, run, and evaluate existing visual bias mitigation methods.

With the increasing awareness of bias in AI systems, it is crucial for researchers to have access to robust tools that facilitate the exploration and development of mitigation approaches. The Visual Bias Mitigator (VB-Mitigator) serves this purpose by offering:

- 🚀 **Implemented Methods**: A collection of established visual bias mitigation methods that can be directly utilized, allowing researchers to replicate and understand their functionality.
- 🔧 **Extensibility**: Researchers can exploit this code-base to develop custom bias mitigation approaches tailored to their specific needs. The framework is designed with flexibility in mind, enabling easy integration of new approaches.
- 📊 **Performance Comparison**: The framework facilitates the performance comparison between custom methods and state-of-the-art. 

The aim of this repository is to facilitate research in the domain of visual bias mitigation. By providing a comprehensive codebase that allows researchers to easily implement and build upon existing methodologies, we encourage the development of new approaches for addressing biases in computer vision tasks.

Quickstart
==========

Get started with Visual Bias Mitigator quickly:

1. Clone the git repository:

   .. code-block:: bash

      git clone https://github.com/gsarridis/vb-mitigator.git

2. Create a virtual environment using either pip or conda and install the required packages:

   .. code-block:: bash

    # create a virtual conda environmnet
    conda create -n vb-mitigator python=3.11

    # activate the environment
    conda activate vb-mitigator

    # install the required packages
    pip install -r requirements.txt

2. Run a sample script:

   .. code-block:: bash
    
    # run BAdd method on UTKFace dataset
    bash ./scripts/utkface/badd/badd.sh

3. Check logs for results and metrics. The output is stored in the `outputs/utkface_baselines/badd` directory.

   **Output Structure:**

   .. code-block::

      ├── outputs
      │   ├── utkface_baselines
      │   │   ├── badd
      │   │   │   ├── logs.csv
      │   │   │   ├── out.log
      │   │   │   ├── best.pth
      │   │   │   ├── latest.pth
      │   │   │   └── train.events


Modules
=======

Methods
-------

Currently, Visual Bias Mitigator offers the following methodologies for mitigating biases in deep learning models:

.. list-table:: 
   :widths: 30 50
   :header-rows: 1

   * - Method
     - Full Name
   * - ERM
     - Empirical Risk Minimization
   * - `GroupDro <https://openreview.net/pdf?id=ryxGuJrFvS>`_
     - Group Distributionally Robust Optimization
   * - `Debian <https://arxiv.org/pdf/2207.10077>`_
     - Debiasing Alternate Networks
   * - `DI <https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Towards_Fairness_in_Visual_Recognition_Effective_Strategies_for_Bias_Mitigation_CVPR_2020_paper.pdf>`_
     - Domain Independent
   * - `EnD <https://openaccess.thecvf.com/content/CVPR2021/papers/Tartaglione_EnD_Entangling_and_Disentangling_Deep_Representations_for_Bias_Correction_CVPR_2021_paper.pdf>`_
     - Entangling and Disentangling
   * - `LfF <https://proceedings.neurips.cc/paper_files/paper/2020/file/eddc3427c5d77843c2253f1e799fe933-Paper.pdf>`_
     - Learning from Failure
   * - `SD <https://arxiv.org/pdf/2011.09468>`_
     - Spectral Decouple
   * - `BB <https://proceedings.neurips.cc/paper/2021/file/de8aa43e5d5fa8536cf23e54244476fa-Paper.pdf>`_
     - Bias Balance
   * - `FLAC <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10737139&casa_token=gSlyiZksMtMAAAAA:2xksz_AVlz1KAlkmIXZ7MI0R4JWDbjdJ_trO9a9UOe6A7etlhNLN8beK1jGQEgWdcoDTbbFhJg&tag=1>`_
     - Fairness Aware Representation Learning
   * - `BAdd <https://arxiv.org/pdf/2408.11439>`_
     - Bias Addition
   * - `Mavias <https://arxiv.org/pdf/2412.06632>`_
     - Mitigate any Visual Bias

To add a new method, follow the instructions in :ref:`Add New Method <add_new_method>`.

Datasets
--------

Currently, Visual Bias Mitigator supports the following datasets:

.. list-table:: 
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Biases
     - Target
   * - `Biased UTKFace <https://github.com/grayhong/bias-contrastive-learning>`_
     - Race or Age
     - Gender
   * - `Biased CelebA <https://github.com/grayhong/bias-contrastive-learning>`_
     - Gender
     - Blonde or Makeup
   * - `Waterbirds <https://github.com/kohpangwei/group_DRO>`_
     - Background (Water or Land)
     - Bird Species
   * - `Biased MNIST <https://github.com/clovaai/rebias>`_
     - Background Colour
     - Digits
   * - `FB-Biased MNIST <https://arxiv.org/pdf/2408.11439>`_
     - Background and Foreground Colours
     - Digits
   * - `ImageNet9 <https://github.com/MadryLab/backgrounds_challenge>`_
     - Unknown
     - Dog, Bird, Vehicle, Reptile, Carnivore, Insect, Instrument, Primate, Fish

Note that all datasets are automatically downloaded and preprocessed by the framework. The only extra requirement is for ImageNet9, which requires manually download the ImageNet dataset beforehand.

To add a custom dataset, follow the instructions in :ref:`Add New Dataset <add_new_dataset>`.


Evaluation Metrics
------------------

Metrics used for fairness evaluation include:

.. list-table:: 
   :widths: 25 50
   :header-rows: 1

   * - Metric
     - Description
   * - `Accuracy <https://github.com/gsarridis/vb-mitigator/blob/main/tools/metrics/acc.py>`_
     - Measures the overall model performance. This metric can be used for balanced test sets.
   * - `Unb/BA/BC Accuracy <https://github.com/gsarridis/vb-mitigator/blob/main/tools/metrics/unb_bc_ba.py>`_
     - Evaluates the model's performance on unbiased, bias-aligned, and bias-conflicting samples.
   * - `WG and Ovr Accuracy <https://github.com/gsarridis/vb-mitigator/blob/main/tools/metrics/wg_ovr.py>`_
     - Evaluates the least-performing subgroup and overall model performance.

To add a new metric, follow the instructions in :ref:`Add New Metric <add_new_metric>`.

Configuration
-------------

The configuration for the Visual Bias Mitigator is managed through a `cfg.py` file utilizing the `YACS <https://github.com/rbgirshick/yacs>`_ (Yet Another Configuration System) library. This design allows for easy customization of various parameters related to experiments, models, datasets, and mitigators. 

**Key Components of the Configuration**


- **Experiment Settings**: Specify project details, experiment names, GPU settings, and random seed for reproducibility.
- **Model Configuration**: Define the type of model to be used (e.g., ResNet) and whether to use pretrained weights.
- **Solver Parameters**: Set up hyperparameters for training, including batch size, learning rate, weight decay, momentum, and training duration.
- **Logging Options**: Configure logging details such as the frequency of TensorBoard updates, checkpoint saving intervals, and logging directory.
- **Dataset Specifications**: Define the dataset type and associated biases, along with any dataset-specific parameters, such as image size and root directories for data.
- **Mitigator Settings**: Configure the bias mitigation approach, specifying the type of mitigator and any additional parameters relevant to specific mitigation methods.

**Example Usage**

The configuration can be accessed and displayed using the `show_cfg` function, which logs the current configuration parameters to help users understand their settings. Here’s a brief example of how this is done:

.. code-block:: python
  
    from yacs.config import CfgNode as CN
    from tools.utils import log_msg

    def show_cfg(cfg, logger):
        dump_cfg = CN()
        dump_cfg.EXPERIMENT = cfg.EXPERIMENT
        dump_cfg.MODEL = cfg.MODEL
        dump_cfg.DATASET = cfg.DATASET
        dump_cfg.MITIGATOR = cfg.MITIGATOR
        dump_cfg.SOLVER = cfg.SOLVER
        dump_cfg.LOG = cfg.LOG
        if cfg.MITIGATOR.TYPE in cfg:
            dump_cfg.update({cfg.MITIGATOR.TYPE: cfg.get(cfg.MITIGATOR.TYPE)})
        log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO", logger)

    
    # CFG example

    CFG = CN()
    CFG.EXPERIMENT = CN()
    CFG.EXPERIMENT.PROJECT = "biased_mnist"
    CFG.EXPERIMENT.NAME = "dev"
    CFG.EXPERIMENT.TAG = "flac"
    CFG.EXPERIMENT.GPU = "cuda:0"
    CFG.EXPERIMENT.SEED = 1

Logs
----

The logging system is designed to store and organize all experiment-related outputs in a structured and accessible format. This setup facilitates tracking, visualization, and reproducibility of results. Each experiment's outputs are stored under the `outputs` directory with the following structure:

   .. code-block::

      ├── outputs
      │   ├── <dataset>_baselines
      │   │   ├── <method>>
      │   │   │   ├── logs.csv
      │   │   │   ├── out.log
      │   │   │   ├── best.pth
      │   │   │   ├── latest.pth
      │   │   │   └── train.events

**Description of Files**

- **`logs.csv`**  
  Contains a tabular record of the training process, including key metrics such as accuracy, loss, and validation performance for each epoch. This format is ideal for further analysis and visualization with external tools.

- **`out.log`**  
  A detailed log file capturing all information about the experiment run, including configuration settings, training progress, validation results, and any warnings or errors encountered.

- **`best.pth`**  
  Stores the model checkpoint with the best performance during training, based on the defined evaluation metric (e.g., validation accuracy).

- **`latest.pth`**  
  Stores the most recent model checkpoint, allowing experiments to resume from the latest training state.

- **`train.events`**  
  TensorBoard event file, storing detailed training and validation statistics for visualization through TensorBoard. This includes metrics such as loss curves and performance trends.

**Benefits**

1. **Readable Logs**: The `out.log` and `logs.csv` files are human-readable, with the latter providing a structured tabular format to facilitate data processing and visualization.  
2. **Visualization**: The `train.events` file integrates seamlessly with TensorBoard for real-time monitoring and analysis.  
3. **Reproducibility**: By saving both `best.pth` and `latest.pth`, users can easily reproduce results or continue training.  
4. **Organized Structure**: The directory hierarchy organizes outputs by experiment and method, ensuring clarity and ease of navigation.

Note that, VB-Mitigator also supports Wandb logging for real-time monitoring and visualization of experiments. To enable Wandb logging, set `LOG.WANDB` to `True` in the configuration file.

Integrate New Components
============================

.. _add_new_method:

Method
---------------

To integrate a new method into the framework, you can use the `BaseTrainer` class provided in `mitigators/base_trainer.py`. This base class defines abstract functions for every step in the training pipeline, allowing you to override and customize the steps for your specific method. Below are the steps to add a new method:

**Step 1: Create a New Method File**

1. Navigate to the `mitigators` directory.  
2. Create a new Python file for your method (e.g., `new_method.py`).  
3. Inherit the `BaseTrainer` class and override the functions where your new method intervenes. For example:

   .. code-block:: python

      from .base_trainer import BaseTrainer

      class NewMethod(BaseTrainer):
        
          def _train_iter(self, batch):
              # Override to define a custom training iteration
              pass


**Step 2: Add the Method to `mitigators/__init__.py`**

1. Open `mitigators/__init__.py`.  
2. Register the new method in the dictionary that maps method names to their respective classes. For example:

   .. code-block:: python

      from .new_method import NewMethod

      method_to_trainer = {
          "erm": ERM,
          "new_method": NewMethod,  # Add your method here
      }

**Step 3: Add Default Hyperparameters to `./configs/cfg.py`**

1. Open `cfg.py`.  
2. Add a new configuration node for your method under `CFG.MITIGATOR`. For example:

   .. code-block:: python

      CFG.MITIGATOR.NEW_METHOD = CN()
      CFG.MITIGATOR.NEW_METHOD.ALPHA = 0.1  # Example hyperparameter
      CFG.MITIGATOR.NEW_METHOD.BETA = 0.5  # Example hyperparameter

**Step 4: Create a Config File for the Experiment**

1. Navigate to the `configs/<dataset>` directory.  
2. Create a new directory for your method (e.g., `new_method`). 
3. Create a YAML file for the experiment (e.g., `dev.yaml`). Define the configuration settings specific to your experiment. For example:

   .. code-block:: yaml

      EXPERIMENT:
        NAME: "new_method"
        TAG: "dev"
        PROJECT: "biased_mnist_baselines"
      DATASET:
        TYPE: "biased_mnist"
        BIASES: ["color"]
      MITIGATOR:
        TYPE: "new_method"
      SOLVER:
        BATCH_SIZE: 64
        EPOCHS: 80
        LR: 0.001
        TYPE: "Adam"
        SCHEDULER:
          LR_DECAY_STAGES: [13, 56]
          LR_DECAY_RATE: 0.1
      MODEL:
        TYPE: "simple_conv"
      METRIC: "acc"

**Step 5: Create a Script for the Experiment**

1. Navigate to the `scripts/<dataset>/` directory.  
2. Create a new directory for your method (e.g., `new_method`).  
3. Inside the directory, create a shell script (e.g., `exp.sh`) to run the experiment. For example:

   .. code-block:: bash

      #!/bin/bash
      python tools/train.py --cfg configs/biased_mnist/new_method/dev.yaml

**Summary**

By following these steps, you will:

- Define your new method in a modular and extensible way.
- Register the method to make it available for use.
- Configure and script an experiment for testing your method.


.. _add_new_dataset:

Dataset
----------------

The framework provides flexibility for adding new datasets. Follow these steps to integrate a new dataset into the framework:

1. **Create the Dataset Class**
   - Implement the dataset class in `datasets/<new_dataset>.py`.
   - In the `__getitem__` method, return a dictionary in the following format:

     .. code-block:: python

        def __getitem__(self, index):
            img, target, bias = (
                self.data[index],
                int(self.targets[index]),
                int(self.biased_targets[index]),
            )
            img = Image.fromarray(img.astype(np.uint8), mode="RGB")

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return {"inputs": img, "targets": target, "<bias_name>"": bias, "index": index}

2. **Update the Dataset Builder**
   - Modify `datasets/builder.py` to include your new dataset.
   - Add the logic to create dataloaders for training, validation, and testing. For Biased MNIST example:

     .. code-block:: python

        train_loader = get_color_mnist(
            cfg.DATASET.BIASED_MNIST.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            data_label_correlation=cfg.DATASET.BIASED_MNIST.CORR,
            n_confusing_labels=9,
            split="train",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )

        val_loader = get_color_mnist(
            cfg.DATASET.BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation=0.1,
            n_confusing_labels=9,
            split="train_val",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )

        test_loader = get_color_mnist(
            cfg.DATASET.BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation=0.1,
            n_confusing_labels=9,
            split="valid",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )

   - Construct a dataset dictionary with the following structure:

     .. code-block:: python

        dataset = {}
        dataset["num_class"] = 10
        dataset["num_groups"] = 10 * 10
        dataset["biases"] = ["background"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["root"] = cfg.DATASET.BIASED_MNIST.ROOT
        dataset["target2name"] = {
            0: "number zero",
            1: "number one",
            2: "number two",
            3: "number three",
            4: "number four",
            5: "number five",
            6: "number six",
            7: "number seven",
            8: "number eight",
            9: "number nine"
        }

3. **Update Configuration**
   - Add any default hyperparameters specific to the new dataset to `cfg.py` under the `CFG.DATASET.<NEW_DATASET>` section.
   - For example:

     .. code-block:: python

        CFG.DATASET.NEW_DATASET = CN()
        CFG.DATASET.NEW_DATASET.ROOT = "./data/new_dataset"
        CFG.DATASET.NEW_DATASET.IMAGE_SIZE = 224
        CFG.DATASET.NEW_DATASET.BIAS = "bias_type"
        CFG.DATASET.NEW_DATASET.TARGET = "target_attribute"

4. **Create Experiment Configuration**
   - Add a new configuration file in the `configs` directory specific to the new dataset.

5. **Add Experiment Script**
   - Create a script for running experiments with the new dataset in `./scripts/<new_dataset>/<method>/exp.sh`.

By following these steps, you can seamlessly integrate new datasets into the framework while maintaining compatibility with its existing structure.


.. _add_new_metric:

Metric
--------------


To integrate a new metric into the framework, follow these steps:

1. **Define the Metric Function**
   - Create a function for the new metric at `tools/metrics/<new_metric.py>`.
   - The function should compute the metric and return a dictionary containing the metric's value(s). Example:

     .. code-block:: python

        def new_metric(data_dict):
            predictions = data_dict["predictions"]
            targets = data_dict["targets"]

            metric_value = some_computation(predictions, targets)
            # you can return multiple values as a dictionary
            return {"new_metric_name": metric_value}

   - Additionally, define a dictionary that specifies optimization preferences (e.g., whether the metric is "high" or "low" for better performance (e.g. performance or error based metrics), and the value wrt which the best model is defined). For example:

     .. code-block:: python

        new_metric_dict = {"best": "high", "performance": "new_metric_name"}

2. **Register the Metric**
   - Add the metric dictionary and function to the `metrics/__init__.py` file by including them in `metrics_dicts` and `get_performance`:

     .. code-block:: python

        from .new_metric import new_metric_dict, new_metric

        metrics_dicts = {
            "acc": acc_dict,
            "unb_bc_ba": unb_bc_ba_dict,
            "wg_ovr": wg_ovr_dict,
            # Register your new metric dictionary
            "new_metric": new_metric_dict,
        }

        get_performance = {
            "acc": acc,
            "unb_bc_ba": unb_bc_ba,
            "wg_ovr": wg_ovr,
            # Register your new metric function
            "new_metric": new_metric,
        }

3. **Select the metrics in cfg**
   - If the new metric requires specific hyperparameters, add them to the configuration file (`cfg.py`) under the appropriate section. For example:

     .. code-block:: yaml

        METRIC: "new_metric"


By following these steps, you can extend the framework with new metrics to suit your evaluation needs.


Contributing
============

We welcome contributions! The ultimate goal of the Visual Bias Mitigator is to provide a comprehensive environment for bias mitigation research in computer vision and thus requires a community effort to intergrate new methods, datasets, and metrics.

To contribute:

1. Fork the repository.
2. Integrate your new feature.
3. Submit a pull request with a detailed description.



FAQ
===

**Q: Can I use this for non-CV tasks?**

A: Currently, Visual Bias Mitigator is optimized for computer vision but can be adapted for other tasks with custom methods.




Changelog
=========

Version 1.0.0:

- Initial release with support for 10 competitive bias mitigation approaches and 6 benchmarks.
- Extensive documentation and tutorials for easy integration and usage.
- Support for custom datasets, metrics, and methods.

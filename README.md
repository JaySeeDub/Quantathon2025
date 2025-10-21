# qBraid  GPU Challenge
[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/JaySeeDub/Quantathon2025/tree/qBraid-GPU-challenge)

# Demo
The 'demo' folder contains a demo jupyter notebook complete with our pipeline and highlights why it necessitates the use of GPUs.

# Installation instructions
1. Create a virtural environment. (Skip this on qBraid.)
2. `pip install -r requirements.txt ` 
    * Note this repo was prepared using python 3.10. It DOES NOT WORK for python 3.13. We did not test any other versions of python.
3. Read through the demo notebook contained located at ./demo/demo.ipynb. This notebook is designed to contain everything relevant to the qBraid GPU challenge from our GPU justification to the actual work we did.

# Problem Description
We have a small and complex dataset of 1000 tornadoes. The goal is to classify the tornadoes into EF categories (tornado strength). This scale is frm EF 0 to EF 6. For our dataset, we will only be using EF 0-3. The main focus will be on identify EF 3 tornadoes correctly so that this information might be used by local authorities to issue weather based emergency alerts to the community.


# Solution
We tackle this problem in multiple ways. First we establish a classical baseline using several different classical AI models. We then use a hybrid quantum-classical approach, and finally we use a fully quantum appraoch: a parameterized quantum circuit (PQC). For each of these three methodologies, we implement both a binary classification and a multi class classification model. The former labels tornadoes as weak or strong where weak is EF 0 and 1, and strong is EF 2 or 3. The latter classifies into strict EF categories, identifying them as either EF 0, 1, 2, or 3.

The hybrid model uses random shadows, a quantum approach to feature engineering, to create additional features. These features are then fed into the classical models.

The PQC is trained on the original dataset, with no feature engineering. Our most promising model is the binary classification PQC. However, this does not distinguish EF 2 tornadoes from EF 3 tornadoes as both are labeled "strong." To circumvent this, we train a Quantum Kernel Estimator (QKE) on the strong tornado data set, slightly more than 10% of the entire dataset, and then feed any strong tornadoes into this QKE, which can differentiate from EF 2 and EF 3 tornadoes. 

## GPU Highlights

Our task and solution would be difficult to solve — and certainly not feasible to complete in a reasonable amount of time — without GPU acceleration. GPUs play a crucial role across every stage of our pipeline, from classical model training to quantum simulation. Below, we highlight the three main areas where GPUs are essential:

* **AI Model Training (Classical and Hybrid Models)**  
  The core of our problem is a machine learning classification task. Training classical models like neural networks or gradient-boosted trees on high-dimensional tornado data involves large-scale linear algebra operations (matrix multiplications, convolutions, etc.), which are highly parallelizable. GPUs allow these computations to occur across thousands of cores simultaneously, drastically reducing training time and enabling faster hyperparameter tuning. This is especially important for our hybrid models, which must repeatedly retrain as new quantum-generated features are introduced from the random shadows step.

* **Random Shadows (Quantum Feature Engineering)**  
  The random shadows method requires estimating the expectation values of multiple observables across a set of quantum circuits. In our implementation, each data point corresponds to a single quantum circuit, and each circuit runs with multiple observables (e.g., 32 per circuit for 1000 circuits total). This results in tens of thousands of circuit executions and expectation value computations. GPUs enable massive parallelization of these simulations, where each circuit–observable pair can be executed concurrently. Without GPUs, this would lead to hours or days of runtime, whereas GPU parallelism allows practical execution times and scalability to larger datasets or more observables per circuit.

* **Quantum Kernel Estimation (Fully Quantum Model)**  
  The Quantum Kernel Estimator (QKE) requires constructing a kernel matrix whose size scales quadratically with the number of input data points. Each element of this matrix corresponds to the overlap between two quantum states, which must be evaluated through repeated circuit simulations. Simulating this process on CPUs would be computationally prohibitive due to the exponential growth of quantum state vectors and the high number of circuit evaluations required. GPUs dramatically accelerate these quantum state simulations by performing vectorized operations and parallel state evolution across many data pairs simultaneously. This makes the QKE viable for non-trivial datasets and allows us to train a differentiable quantum model that distinguishes EF2 from EF3 tornadoes efficiently.

* **End-to-End Quantum–Classical Integration**  
  Finally, our workflow involves multiple stages of computation — classical preprocessing, quantum circuit simulation, feature extraction, and retraining — each of which benefits from GPU acceleration. The tight feedback loop between classical and quantum components demands rapid data transfer and computation to remain interactive and scalable. GPUs provide the throughput necessary for this end-to-end integration, enabling our system to move seamlessly from raw tornado data to high-fidelity EF category predictions.

In short, GPUs are not an optional enhancement for our project — they are the foundation that makes our hybrid and fully quantum models computationally feasible. Without GPU acceleration, our entire workflow would be bottlenecked at every stage, preventing meaningful experimentation or real-time model refinement.



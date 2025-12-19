# PredictiveCodingTutorial  

This repository contains tutorial-style notebooks for the Lecture Notes article **"Bio-Inspired Artificial Neural Networks Based on Predictive Coding."**  

## Folder Structure: `./PC_BP_Gradients`  

This folder contains a notebook demonstrating how **backpropagation gradients can be approximated using predictive coding**, as discussed in the Lecture Notes and in the following publication: [An Approximation of the Error Backpropagation Algorithm in Predictive Coding Networks](https://direct.mit.edu/neco/article/29/5/1229/8261/An-Approximation-of-the-Error-Backpropagation)  

## Folder Structure: `./classification`

This folder contains classification tasks. It includes a `models` folder with network architectures, a `FLOPs` file analyzing resource usage, and a `comparison` file comparing model performance when trained with BP or PC.

## Folder Structure: `./compression`

This folder contains autoencoder models for compression. It includes a `models` folder with network architectures, a `FLOPs` file analyzing resource usage, and a `comparison` file comparing model performance when trained with BP or PC.

## Folder Structure: `./images`

This folder contains a schematic of the architectures used, showing the differences between PC and BP networks for both classification and compression/generation settings. Additionally, the file 'pseudocodes.png' provides the pseudocode for both algorithms.


## Important Notes  

Predictive coding models **must maintain a consistent batch size throughout training** because neurons are treated as `nn.Parameters`, just like network weights. As a result, their shape cannot be modified during training. 

When computing the Free Energy gradient using self.energy.backward(), gradients are calculated for both neural activities and weights. After running T updates of neural activity during the predictive coding inference phase, the neural activity used by PyTorch to compute the weight gradients corresponds to the state at time step T-1. This happens because PyTorch computes gradients based on the saved variables at the moment of the backward call, and any subsequent updates to those variables do not affect the computed gradients.
To ensure the weight gradients use neural activities optimized over all T steps, the inference phase should be run for T+1 steps before performing the weight update. This guarantees that the neural activity state PyTorch uses matches the fully optimized state after T updates.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## References on Alternative Local Learning Rules
- **Leader-Follower Neural Networks with Local Error Signals Inspired by Complex Collectives** — [arXiv](https://arxiv.org/abs/2310.07885)  
- **Unlocking Deep Learning: A BP-Free Approach for Parallel Block-Wise Training of Neural Networks** — [IEEE Xplore](https://ieeexplore-ieee-org.tudelft.idm.oclc.org/stamp/stamp.jsp?tp=&arnumber=10447377)  
- **Learning Without Feedback: Fixed Random Learning Signals Allow for Feedforward Training of Deep Neural Networks** — [Frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.629892/full)  
- **Direct Feedback Alignment Provides Learning in Deep Neural Networks** — [arXiv](https://arxiv.org/pdf/1609.01596)  
- **Hebbian Deep Learning Without Feedback** — [arXiv](https://arxiv.org/pdf/2209.11883)  

## References on Alternative Optimization Algorithms
- **Levenberg-Marquardt** — [PDF](https://sites.cs.ucsb.edu/~yfwang/courses/cs290i_mvg/pdf/LMA.pdf)  
- **Levenberg–Marquardt Training** — [Taylor & Francis](https://www.taylorfrancis.com/chapters/edit/10.1201/9781315218427-12/levenberg%E2%80%93marquardt-training-hao-yu-bogdan-wilamowski)  

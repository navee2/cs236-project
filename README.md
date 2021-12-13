# CS236 Project
This repo contains code for CS236 project for Fall - 2021 Following is a copy of original Project proposal.
##Problem Statement
Map the complex information of NN parameters to an autoencoder which
can be used to generate new Neural networks. A set of Neural networks trained on a dataset
will be jointly encoded and decoded using an autoencoder.
##Previous/Related Work
Task2Vec[1] provides a way to represent visual classification tasks
which can be used to reason about the nature of those tasks and their relations. It separates NN
model into probe network and classifier. An embedding of task is created using probe network.
There are multiple applications of this embedding such as identifying similar tasks with
similarity measure, transfer learning by using transfer distance from embeddings etc.
<\br>Neural network compression for noisy storage devices [2] provides experiments on compression
of Neural Network. Compared to conventional error-free digital storage, this method has the
potential to reduce the memory size by one order of magnitude, without significantly
compromising the stored model’s accuracy. Neural network compression has major applications
in resource-constrained devices.
Dataset: MNIST will be our base dataset which is a database of handwritten digits. There are
over 60,000 examples in this dataset. MNIST is a good starting point for a proof-of-concept
experiment for complex meta-learning task as training it is easier compared to more complex
datasets. We shall generate multiple Neural networks (MLP) using it which will be dataset for
training Autoencoder
## Methodology and Experiments:
Both non-generative and generative autoencoders will be used to create an encoding for MLP.
We shall look into MLP reconstruction and analyze the encodings. Following are expected
experiments in this endeavor:

**Experiment 1:** build an autoencoder to reproduce multiple MLPs trained on MNIST. These
MLPs will be trained using differing initializations. Structure of MLP will be simple for initial
encoding.

**Experiment 2:** Analysis of encoding-decoding errors and trying out different architectures for
autoencoder

**Experiment 3:** Generalize methodology to more complex MLP structure

**Experiment 4:** Experiment with generative autoencoders to generate good MLPs for a given
problem
##Evaluation
 Quantitatively we shall check whether Autoencoder is doing a good job at
reproducing MLP using accuracy on original dataset, distance from original MLP and Fisher
information matrix. Qualitatively, we shall look into decision boundaries, intermediate layers of
autoencoders, size of encoding to assess quality of compression

##References
1. Alessandro A et al, Task2Vec: Task Embedding for Meta-Learning, arXiv:1902.03545
2. Berivan Isik et al, Neural Network Compression for Noisy Storage Devices
arXiv:2102.07725
# Report on 2018/06/12
## Implementing the code
- Still some problem with implement the 64 hidden units, not sure how to do it.
- Early stopping implementation is still unclear.

## CUDA initializing
- Managed to make it work on the GPU.
- Unsure if everything is working because of the problems with the hardware.

## PyTorch
- Implementation of the Neural Network.
- Using the parameters determined in the paper.

## Skorch
- Used to wrap the code of the neural network to simplefy the implementation of the training.
- Using the callback functions to generate early stopping if possible.
- Used for the implementation of GridSearch Cross Validation.

## Problems in working on the GPU computer
- Data won't load. Commmand data = pd.read_csv does not work. The problem might come from Anaconda. I might need to freshly redownload everything.
- I have a problem using skorch (package to implement gridsearch in an easy way, wraps the code of the neural network).

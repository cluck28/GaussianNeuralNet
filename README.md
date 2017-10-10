# GaussianNeuralNet
Simple neural network to identify the width of a Gaussian distribution

I include files to generate training, test, and validation data (GaussianGenerator), to fit the gaussian distributions (for performance comparison) (Gaussian Fit), to train the network (no hidden layers currently) (logistic), to re-train a network with preset initial conditions (logistic_pretrain), to test the network's performance (TestAnalysis), and to unpack the parameters for easy analysis(UnpackParams)

I include the most recently trained net for Gaussians with random noise (~40% amplitude)

The most recent branch has files with fringes (variable amplitude and offset in each image) and Gaussian noise. This is the most realistic to the images in experiment.

# Tree species classification based on MLP-Mixer

A deep learning model for the automatic classification of tree species from point cloud data.

The point cloud data: >17k individual trees from 33 tree species captured with TLS, MLS and ULS.

The data are divided into training and validation set. The training set consits of 90% of the data, whereas the rest 10% comprise the validation set.
The split is done so that the training data contains a representative number of trees with various diameters from all three sensors.

The PointMixer network is trained with the following settings and hyperparameters:
- Input points: 2048, selected from the original point cloud using the Furthest Point Sampling
- 42 batch size
- Initial learning rate: 0.1
- Cosine annealing schedule with minumum learning rate of 0.005
- Training epochs: 300/400

The best model was chosen based on the performance on the validation set.

Confusion matrix on the validation set:


Metrics computed on the validation set:
| Overall accuracy | Precision | Recall | F1-score |
| ------------- | ------------- | ------------- | ------------- |
| 0.76 | 0.75 | 0.75 | 0.76

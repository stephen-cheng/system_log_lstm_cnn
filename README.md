## System Log with LSTM & CNN

Train and predict log message with deep learning methods.

### Paper

[Deep Convolutional Neural Networks for Log Event Classification on Distributed Cluster Systems](https://ieeexplore.ieee.org/document/8622611)

With the widespread development of cloud computing, cluster systems are becoming increasingly complex, system logs is an universal and effective approach for automatic system management and troubleshooting. Log event classification as an effective preprocessing method for log analysis, which is helpful for system administrators to locate or predict components' have errors or failures. In this paper, we design and implement an automatic log classification system based on deep CNN (Convolutional Neural Network) models, and take advantage of the feature engineering and learning algorithm to improve classification performance. First, in the feature engineering step, to address the problem of that the original unstructured event logs are unsuitable for numerical calculation in deep CNN models, we propose a novel and effective log preprocessing method, which include building categories dictionary libraries, filtering abundant information, generating numerical semantic feature vectors by calculating and combining the semantic similarity values for filtered log events. Additionally, in the learning step, we measure a series of deep CNN algorithms with varied hyper-parameter combinations by using standard evaluation metrics, and the results of our study reveal the advantages and potential capabilities of the proposed deep CNN models for log classification tasks on cluster systems. The optimal classification precision of our approach is 98.14%, which surpasses the popular traditional machine learning methods, and it can also be applied to other large-scale system logs with good accuracy. Just like the experiment results, different choices of learning algorithm do result in performance numbers varying, and subsequently careful feature engineering enables promoting performances, thus both of approaches contribute to best learning model finding.

### Parameters

- Deep CNN Model:

- Layers: 2+2 3+2 5+2 5+3 7+2 7+3 (conv-layer + fully-conn layer)

- Learning rate: 0.1, 0.01, 0.001, 0.0001

- Hidden layer size: 16, 32, 64, 128

- Dropout: 0.25, 0.5, 0.75, 1.0

### Results Comparison 

[Figure](https://raw.githubusercontent.com/steven-cheng-com/system_log_lstm_cnn/master/code/lstm_cnn/figure/figure6.png)


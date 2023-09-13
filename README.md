# Traffic Flow Prediction System
COS30018 - Intelligent Systems

## Getting Started

These instructions will explain the process of getting the system up and running on your local machine.

### Prerequisites

Python 3.7
```
keras
pandas
scikit-learn
tensorflow
virtualenv
```

### Installing

Download/clone the repository into a folder on your computer.
```
https://github.com/MagmaArcade/Traffic-Flow-Prediction.git
```

## Train the model

**Run command below to train the model:**

```
python train.py --model {model_name}
```

You can choose "lstm", "gru" or "saes" as arguments. The ```.h5``` weight file was saved at model folder.


**Run command below to run the program:**

```
python main.py
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | - | - | - | - | - | - |
| GRU | - | - | - | - | - | - |
| SAEs | - | - | - | - | - | - |

![evaluate](/images/eva.png) to be added


## Acknowledgments

* **xiaochus** - *Base code* - [Traffic Flow Prediction](https://github.com/xiaochus/TrafficFlowPrediction)

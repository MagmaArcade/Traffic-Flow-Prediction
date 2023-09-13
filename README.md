# Traffic Flow Prediction System
COS30018 - Intelligent Systems

## Getting Started

These instructions will explain the process of getting the system up and running on your local machine.

### Prerequisites

The following Python 3.7 Libaries are required to run the program. for windowns use pip install, for mac use ---
```
keras
pandas
scikit-learn
tensorflow
virtualenv
pydot-ng
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

2 epoch trial
| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | 17.97 | 613.68 | 24.77 | 74.17% | 0.91 | 0.9245 |
| GRU | 19.35 | 679.71 | 26.07 | 103.312780% | 0.90 | 0.9099 |
| SAEs | 22.20 | 954.53 | 30.89 | 102.27% | 0.87 | 0.8756 |

600 epoch trial
| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | - | - | - | - | - | - |
| GRU | - | - | - | - | - | - |
| SAEs | - | - | - | - | - | - |

![evaluate](/images/eva.png) to be added


## Acknowledgments

* **xiaochus** - *Base code* - [Traffic Flow Prediction](https://github.com/xiaochus/TrafficFlowPrediction)

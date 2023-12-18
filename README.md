# Traffic Flow Prediction

## Installation Prerequisites

Before deploying the project, ensure the following libraries are installed using the specified commands:

- **Keras:** `pip install keras`
- **Pandas:** `pip install pandas`
- **Scikit:** `pip install scikit-learn`
- **Tensor Flow:** `pip install tensorflow`
- **Virtual Environment:** `pip install virtualenv`
- **Pyglet version 1.5.11:** `pip install pyglet==1.5.11`
- **Python Open GL:** `pip install PyOpenGL`

## Installing Code

Download or clone the repository into a folder on your computer using the following link:
[https://github.com/MagmaArcade/Traffic-Flow-Prediction.git](https://github.com/MagmaArcade/Traffic-Flow-Prediction.git)

## Training the Model

To enable traffic prediction, train the learning models by running the following command for each training model, replacing `{model_name}` with options: "lstm," "gru," or "saes."
`python train.py --model {model_name}`

## Running the Program
After training the models, run the program using the following command:
`python gui.py`

Arguments for gui.py
--destination
--origin
--time
--model

Example:
`python gui.py --destination=3001 --origin=4201 --time=14:30 --model=gru`

Alternatively, you can run the following command to get flow prediction for one node:
`python task1-2test.py`

Additional Arguments for task1-2test.py
--scats
--direction
--time
--model
Example:
`python task1-2test.py --scats=3001 --direction=NE --time=22:15 --model=lstm`


## GUI Interaction

The program opens a GUI menu awaiting user input. Use the following keys to interact with the program:

Press the Space key to calculate routes.
Press the Q key to cycle through calculated routes.
Press Tab to toggle between selecting a new Origin or Destination.
Left-click on a node to select a new Origin or Destination.
Press the W key to cycle through models.






## Acknowledgments

* **xiaochus** - *Base code* - [Traffic Flow Prediction](https://github.com/xiaochus/TrafficFlowPrediction)

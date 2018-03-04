# NLP-domain-specific-assistant
NLP based domain specific assistant

A NLP based ML assistant.

Details of project:
- Tensorflow 1.5
- Bidirectional encoder
- Attention
- Beam search decoder for Inference
- Data set - Cornell movie corpus

Setup/Requirements:
- Python 3 - `brew install python3`
- Tensforflow 1.5 installation using - https://www.tensorflow.org/install/
  - Activating Tensorflow virtual env - 
    - `$ cd targetDirectory`
    - `$ source ./bin/activate`
- Numpy

To Execute the project:
- Training: Change the flag in the main.py file to Training
- Prediction:
  - Change the flag to input_file for Prediction using input file prediction_input.
  - Change the flag to command_line for Prediction using command line.

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
      `$ cd targetDirectory`
      `$ source ./bin/activate`
  - Numpy

To Execute the project:

  - Training: `python3 main.py training`
  - Testing: `python3 main.py testing` - used for the automatic evaluation of the responses using BLEU
  - Prediction:
    - Input file mode - `python3 main.py inference file` OR `python3 main.py inference`
    - Command line mode - `python3 main.py inference command`
    
   - Data preparation (if needed): 
      - movie_conversation.txt and movie_lines.txt should be present under `/Data`
      - Run `python3 data_preparation.py`

To run TensorBoard:

  - Activate tensorflow virtual environment
  - `tensorboard --logdir=locationOfmodelDirectory`


Add a directory for model:
  - `/model/seq2seq` to store checkpoints


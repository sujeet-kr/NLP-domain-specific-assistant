import tensorflow as tf
import logging

import Seq2Seq as s2s


def main():
    print("STARTED \n")
    tf.logging._logger.setLevel(logging.INFO)
    #FOR TRAINING
    # s2s.train_seq2seq('Data/final_question_file', 'Data/final_answer_file', 'Data/vocab_map', 'model/seq2seq')
    #FOR PREDICTION     ---  'input_file'/'command_line'
    s2s.predict_seq2seq('Data/prediction_input','Data/vocab_map', 'model/seq2seq', 'input_file')
    # s2s.predict_seq2seq('Data/prediction_input','Data/vocab_map', 'model/seq2seq', 'command_line')

    print("FINISHED \n")


if __name__ == "__main__":
    main()
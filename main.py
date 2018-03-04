import tensorflow as tf
import logging
import sys

import Seq2Seq as s2s


def main():
    print("STARTED \n")

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        raise ValueError("Main mode option not provided")

    tf.logging._logger.setLevel(logging.INFO)
    if mode.upper() == "TRAINING":
        #FOR TRAINING
        print("Training Started")
        s2s.train_seq2seq('Data/final_question_file', 'Data/final_answer_file', 'Data/vocab_map', 'model/seq2seq')
    elif mode.upper() == "INFERENCE":
        if len(sys.argv) > 2:
            pass
        else:
            sys.argv.append("COMMAND")
        #FOR PREDICTION     ---  'input_file'/'command_line'
        infer_mode = sys.argv[2]
        if infer_mode.upper() == "FILE":
            print("Entered Inference File Mode")
            s2s.predict_seq2seq('Data/prediction_input','Data/vocab_map', 'model/seq2seq', 'input_file')
        elif infer_mode.upper() == "COMMAND":
            print("Entered Inference Command Mode")
            command_line_input = input("Question: ")
            ans = s2s.predict_seq2seq('Data/prediction_input','Data/vocab_map', 'model/seq2seq', 'command_line',
                                      command_line_input)
            print("\nQuestion: ", command_line_input)
            print("\nAnswer: ", ans.replace('<EOS>',''))
        else:
            raise ValueError("Correct Inference mode (FILE/COMMAND) was not supplied")
    else:
        raise ValueError("Correct Main mode (Training/Inference) was not supplied")

    print("\nFINISHED \n")


if __name__ == "__main__":
    main()
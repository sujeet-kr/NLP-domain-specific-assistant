import numpy as np
import re

min_line_length = 2 # Minimum number of words required to be in a Question/Answer for consideration in training
max_line_length = 20 # Minimum number of words allowed to be in a Question/Answer for consideration in training
min_number_of_usage = 1 # minumum number of usage in the questions/answers to consider being included in vocab

# <PAD>=8093
# <EOS>=8094
# <UNK>=8095
# <GO>=8096


def main_prepare_data():
    print("Data preparation Started")

    lines, conv_lines = load_data_from_file()
    clean_questions, clean_answers = create_questions_and_answers_array_tmp(lines, conv_lines )
    # Remove questions and answers that are shorter than 2 words and longer than 20 words.
    short_questions, short_answers = create_final_question_answer_data(clean_questions, clean_answers)

    length_questions = len(short_questions)
    length_answers = len(short_answers)
    #Write the short questions and answers to file
    if(length_questions==length_answers):
        write_lines_to_file('Data/input_text', short_questions)
        write_lines_to_file('Data/output_text', short_answers)

        print("Total number of Questions is ", length_questions)
        print("Total number of Answers is", length_answers)
    else:
        print("ERR:: Questions have ", length_questions, " while Answers have ", length_answers)

    # Create a dictionary for the number of times all the words are used in short questions and short answers
    dict_word_usage = create_dictionary_word_usage(short_questions, short_answers)
    print("Total number of words started with in dictionary ", len(dict_word_usage))
    #Create a common vocab for questions and answers along with the special codes
    vocab_words_to_int = vocab_from_word_to_emb_without_rare_word(dict_word_usage, min_number_of_usage)
    # questions_int_to_vocab, answers_int_to_vocab = vocab_decode_from_emb_to_words(questions_vocab_to_int, answers_vocab_to_int)
    write_dict_to_file(vocab_words_to_int, 'Data/vocab_map')
    print("Total number of words finally in dictionary ", len(vocab_words_to_int))

    #sort the questions and answers based on the number of words in the line
    sorted_questions, sorted_answers = sort_question_answers_based_on_number_of_words(
                                                                short_questions, short_answers, max_line_length)

    write_lines_to_file("Data/final_question_file", sorted_questions)
    write_lines_to_file("Data/final_answer_file", sorted_answers)


def write_lines_to_file(filename,list_of_lines):
    with open(filename,'w') as file_to_write:
        for i in range(len(list_of_lines)):
            file_to_write.write(list_of_lines[i] + "\n")



def readlineFromFile(filename):
    file = open(filename, "r")
    # print(file.readline())
    for i in range(10):
        print(file.readline())

def load_data_from_file():
    lines = open('Data/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conv_lines = open('Data/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    return lines, conv_lines

def create_final_question_answer_data(clean_questions, clean_answers):

    # Filter out the questions that are too short/long
    short_questions_temp = []
    short_answers_temp = []

    i = 0
    for question in clean_questions:
        if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
            short_questions_temp.append(question)
            short_answers_temp.append(clean_answers[i])
        i += 1

    # Filter out the answers that are too short/long
    short_questions = []
    short_answers = []

    i = 0
    for answer in short_answers_temp:
        if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
            short_answers.append(answer)
            short_questions.append(short_questions_temp[i])
        i += 1

    return short_questions, short_answers




def create_questions_and_answers_array_tmp(lines, conv_lines):
    # Create a dictionary to map each line's id with its text
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

    # Create a list of all of the conversations' lines' ids.
    convs = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(_line.split(','))

    # Sort the sentences into questions (inputs) and answers (targets)
    questions = []
    answers = []

    for conv in convs:
        for i in range(len(conv) - 1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i + 1]])

    # Clean the data
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))

    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    return clean_questions, clean_answers



def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"temme", "tell me", text)
    text = re.sub(r"gimme", "give me", text)
    text = re.sub(r"howz", "how is", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"[-()\"#[\]/@;:<>{}`+=~|.!?,]", "", text)

    return text


def create_dictionary_word_usage(short_questions, short_answers):
    # Create a dictionary for the frequency of the vocabulary
    vocab = {}
    for question in short_questions:
        for word in question.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for answer in short_answers:
        for word in answer.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab


def vocab_from_word_to_emb_without_rare_word(dict_word_usage, min_number_of_usage):

    vocab_words_to_int = {}


    vocab_words_to_int['<GO>'] = 0
    vocab_words_to_int['<EOS>'] = 1
    vocab_words_to_int['<UNK>'] = 2
    vocab_words_to_int['<PAD>'] = 3

    word_num = 4
    for word, count in dict_word_usage.items():
        if count >= min_number_of_usage:
            vocab_words_to_int[word] = word_num
            word_num += 1

    return vocab_words_to_int


def vocab_decode_from_emb_to_words(questions_vocab_to_int, answers_vocab_to_int):
    # Create dictionaries to map the unique integers to their respective words.
    # i.e. an inverse dictionary for vocab_to_int.
    questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
    answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}
    return questions_int_to_vocab, answers_int_to_vocab

def write_dict_to_file(dict_to_write, file_to_write):
    with open(file_to_write,'w') as file_to:
        for key,val in dict_to_write.items():
            file_to.write(str(key) + "=" + str(val) + "\n")


def convert_input_to_embeddings(input_list, vocab_to_int):
    # Convert the text to integers.
    # Replace any words that are not in the respective vocabulary with <UNK>
    output_int = []
    for input_line in input_list:
        ints = []
        for word in input_line.split():
            if word not in vocab_to_int:
                ints.append(vocab_to_int['<UNK>'])
            else:
                ints.append(vocab_to_int[word])
        output_int.append(ints)

    return output_int



def write_question_answer_embeddings_to_file(embeddings_list, file_name):
    with open(file_name,'w') as file_to:
        for lines in embeddings_list:
            for words in lines:
                file_to.write(str(words) + " ")
            file_to.write("\n")




def sort_question_answers_based_on_number_of_words(questions, answers, max_line_length):
    # Sort questions and answers by the length of questions.
    # This will reduce the amount of padding during training
    # Which should speed up training and help to reduce the loss

    sorted_questions = []
    sorted_answers = []

    for length in range(min_line_length, max_line_length):
        for i, ques in enumerate(questions):
            ques_tmp = ques.split(" ")
            if len(ques_tmp) == length:
                sorted_questions.append(questions[i])
                sorted_answers.append(answers[i])

    return sorted_questions, sorted_answers


if __name__ == "__main__":
    main_prepare_data()
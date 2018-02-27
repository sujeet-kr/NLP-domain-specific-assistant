import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import Seq2Seq as s2s

app = Flask(__name__)
api = Api(app)

#train
train_parser = reqparse.RequestParser()
train_parser.add_argument('dataset_url', type=str, required=True, help='url to dataset is required', location='json')
train_parser.add_argument('id', type=int, required=True, location='json')
train_parser.add_argument('dataset_id', type=int, required=True, location='json')
#predict
predict_parser = reqparse.RequestParser()
predict_parser.add_argument('question', type=str, required=True, location='json')
predict_parser.add_argument('id', type=int, required=True, location='json')
predict_parser.add_argument('dataset_id', type=int, required=True, location='json')

class Train(Resource):

    def post(self):
        args = train_parser.parse_args()
        dataset_url = args.dataset_url
        ## do trainging stuff here
        return 200

class Predict(Resource):

    def post(self):
        args = predict_parser.parse_args()
        input_question = args.question
        input_file = "../Data/prediction_api_input"
        file = open(input_file, "w")
        file.write(input_question)
        file.close()
        return s2s.predict_seq2seq(input_file, '../Data/vocab_map', '../model/seq2seq', 'API')


api.add_resource(Predict, '/predict')
api.add_resource(Train, '/train')

if __name__ == '__main__':
    app.run(debug=True)
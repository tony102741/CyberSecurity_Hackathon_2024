# import module
from flask import Flask, request, jsonify
import re
import json
import os

# flask
app = Flask(__name__)

# URL extractor
def URL_Parsing(Message):
    URL_Parser = re.compile(r'(https?:\/\/)?(\w+\.){1,255}(\w+)\/?(\w+)?')
    result = URL_Parser.findall(Message)
    count = len(result)
    return (count, result)

# Phone number extractor
def Phone_Number_Parsing(Message):
    result = list()
    Phone_Number_Parser_0 = re.compile(r'\d{2,3}-?\d{3,4}-?\d{4}')
    result = result + Phone_Number_Parser_0.findall(Message)
    Phone_Number_Parser_1 = re.compile(r'\d{4}-\d{4}')
    result = result + Phone_Number_Parser_1.findall(Message)
    count = len(result)
    return (count, result)

# Load rule definition
# Excluding special directories for Unix-based operating systems
patern_file_list = os.listdir("./Pattern_Set")
if '.' in patern_file_list:
    patern_file_list.remove(".")
if '..' in patern_file_list:
    patern_file_list.remove("..")

# http post API
@app.route('/', methods=['POST'])
def check():
    if request.method == 'POST':
        # read request json
        input_json = request.get_json()
        
    # Load and compare each pattern file
    for file_name in patern_file_list:
        score = 1
        with open(os.path.join("./", "Pattern_Set", file_name), "r", encoding="UTF-8") as file:
            patern_json = json.load(file)
            if patern_json["URL_Check"][0]:
                url_count, url_list = URL_Parsing(input_json["Message"])
                score *= url_count if url_count else 1 * patern_json["URL_Check"][1]
                score += url_count * patern_json["URL_Check"][2]
            if patern_json["Phone_Number_Check"][0]:
                phone_number_count, phone_number_list = Phone_Number_Parsing(input_json["Message"])
                score *= phone_number_count if phone_number_count else 1 * patern_json["Phone_Number_Check"][1]
                score += phone_number_count * patern_json["Phone_Number_Check"][2]
            title = patern_json["Title"]
            threshold = patern_json["Threshold"]
            for keyword, weight, bias in patern_json["KeywordSet"]:
                if keyword in input_json["Message"]:
                    score *= weight
                    score += bias
                    if score > threshold:
                        return jsonify({'result' : title})
    return jsonify({'result' : "None"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
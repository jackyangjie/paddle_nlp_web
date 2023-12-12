import logging as logger

from request_handle import BaseRequest,url,headers,ChatRequest
import requests,json


glob_labels_path = "/mnt/data/paddlenlp/text_class/label.txt"

glob_labels = []
def labels_init():
    with open(glob_labels_path,"r") as file:
         global glob_labels
         for i, line in enumerate(file.readlines()):
             glob_labels.append(line.strip())

class TextClassHandle(BaseRequest):

    path_name = "/models/cls_hierarchical"
    def support(self, request: ChatRequest):
        if request.prompt is None:
            return True

    def send_request(self, request: ChatRequest):
        prompt = request.prompt
        data = {
            "data": {
                "text": [request.query],
            },
            "parameters": {"max_seq_len": 512, "batch_size": 8, "prob_limit": 0.5},
        }
        r = requests.post(url=url+self.path_name, headers=headers, data=json.dumps(data))
        response = self.format_response(r.text)
        logger.info("文本内容:{}".format(request.query))
        logger.info("分类结果:{},标签转换结果:{}".format(r.text,response))
        logger.info("-----------------")
        return response
    def format_response(self,response):
        response_json = json.loads(response)
        result_json = response_json["result"]
        labels = result_json["label"]
        confidences = result_json["confidence"]
        global glob_labels
        chan_label = []
        if len(labels) <= 0:
            return chan_label
        for label in labels:
            if len(label) <= 0:
                return chan_label
            for l in label:
                chan_label.append(glob_labels[l])

        return chan_label




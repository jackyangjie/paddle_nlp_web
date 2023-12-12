import json
import logging as logger
import requests
from request_handle import BaseRequest,url,headers,ChatRequest

class EntityExtHandle(BaseRequest):

    path_name="/chat"

    default_schema = ["人物","组织"]

    requestParam = {
        "data": {
            "text": [],
        },
        "parameters": {
            "schema": default_schema
        }
    }

    def support(self, request: ChatRequest):
        if request.prompt is not None:
            return True

    def send_request(self,request: ChatRequest):
        text = request.query
        schema = request.prompt
        if text is None:
            return
        self.requestParam["data"]["text"]=[text]
        if schema != None:
            self.requestParam["parameters"]["schema"]=json.loads(schema)
        r = requests.post(url=url+self.path_name, headers=headers, data=json.dumps(self.requestParam))
        datas = json.loads(r.text)
        logger.info("提取结果:{}".format(datas))
        response = self.format_response(datas["result"])
        return response
    def format_response(self,result):
        data=[]
        for r in result:
            if len(r) == 0:
                continue
            for key,value in r.items():
                entry = {key:[]}
                for t in value:
                    if t['text'] == '':
                        continue
                    entry[key].append(t['text'])
                    #去掉重复出现的实体
                    entry[key]=list(dict.fromkeys(entry[key]))
                data.append(entry)
        return data

# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from paddlenlp.server import BaseTaskflowHandler
from paddlenlp.utils.log import logger

RESP_SUCCESS = "SUCCESS"
RESP_FAILURE = "FAILURE"
class TrsTaskflowHandler(BaseTaskflowHandler):
    def __init__(self):
        self._name = "trs_taskflow_handler"

    @classmethod
    def process(cls, predictor, data, parameters):
        if data is None:
            return {}
        text = None
        if len(data) > 0:
            text = data
        else:
            return {}
        if parameters is not None:
            schema = parameters.split(",")
            predictor.set_schema(schema)
        try:
            result = predictor(text)
            format_result = format_response(result)
            return format_result
        except Exception as e:
            return {"result": RESP_FAILURE, "response": e, "history": []}


def format_response(result,schema):
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
    return {"result": RESP_SUCCESS, "response": data, "history":[]}


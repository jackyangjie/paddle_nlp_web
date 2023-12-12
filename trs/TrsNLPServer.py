from paddlenlp import Taskflow,SimpleServer
from paddlenlp.server import CustomModelHandler, MultiLabelClassificationPostHandler,ERNIEMHandler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# The schema changed to your defined schema
schema = ["人物","组织"]
# The task path changed to your best model path
uie = Taskflow("information_extraction", schema=schema, task_path="/home/PaddleNLP/model_zoo/uie/checkpoint/model_best/")
# If you want to define the finetuned uie service
# trsApp = TrsSimpleServer()
# trsHandler = TrsTaskflowHandler()
#trsApp.register_taskflow("chat", uie,trsHandler)
trsApp = SimpleServer()
#实体提取
trsApp.register_taskflow("chat",uie)


#文本分类
trsApp.register(
    "models/cls_hierarchical",
    model_path="/mnt/data/paddlenlp/text_class/",
    # tokenizer_name="ernie-3.0-medium-zh",
    tokenizer_name="ernie-3.0-xbase-zh",
    model_handler=CustomModelHandler,
    # model_handler=ERNIEMHandler,
    post_handler=MultiLabelClassificationPostHandler,
    device_id=3
)



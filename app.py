import ast
import urllib3

from flask import Flask, request
from quntization import Quntization

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class QuntizationApp:
    def __init__(self, port, ip_address, revision):
        self.app = Flask(__name__)
        self.port = port

    def qunt_run(self):
        payload = ast.literal_eval(request.get_data().decode("UTF-8"))
        qt = Quntization(model_path = payload['model_path'],
                    device_type = payload['device_type'],
                    model_type = payload['model_type'],
                    network_name = payload['network_name'],  # mobilenetv3_small_075 고정
                    model_name = payload['model_name'],
                    version = payload['version'])
        qt.run(True)

        return 'success'
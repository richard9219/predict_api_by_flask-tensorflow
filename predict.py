import flask
import pandas as pd
from tensorflow.keras.models import load_model

# 实例化 flask
app = flask.Flask(__name__)


# 加载模型
model = load_model('model.h5')

# 将预测函数定义为一个端点
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # 若发现参数，则返回预测值
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        print(x)
        data["prediction"] = str(model.predict(x)[0][0])
        data["success"] = True


    # 返回Jason格式的响应
    return flask.jsonify(data)

if __name__ == '__main__':

    # 启动Flask应用程序，允许远程连接
    app.run(host='127.0.0.1',port='5555')
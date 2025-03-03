from flask import Flask, request, render_template,jsonify,send_file
import pickle
import pandas as pd
from io import BytesIO


app = Flask(__name__)


with open('./templates/lasso_stacking.pkl', 'rb') as f:
    model = pickle.load(f)
def predict_risk(features):
    # 预测模型返回结果（0表示存活，1表示死亡）
    prediction = model.predict([features])
    if prediction[0] == 1:
        return "Negative"
    else:
        return "Positive"
    
@app.route('/')
def hello_world():
    return render_template("cadpre.html")

@app.route('/download_template')
def download_template():
    return send_file("./static/template_patients.csv", as_attachment=True, download_name="template_patients.csv")


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # 从内存中读取文件内容
    file_content = file.read()
    
    # 根据文件类型处理数据
    if file.filename.endswith('.csv'):
        mdata = pd.read_csv(BytesIO(file_content))
    elif file.filename.endswith('.xlsx'):
        mdata = pd.read_excel(BytesIO(file_content))
    else:
        return jsonify({"error": "Invalid file format"})
    
    # 进行预测
    predictions = []
    for index, row in mdata.iterrows():
        features = [
            row['gcs'],  # GCS Score
            row['CKD_Stage'],  # CKD Stage
            row['height'],  # Height
            row['anchor_age'],  # Age
            row['Platelet Count (K/uL)'],  # Platelet Count
            row['Bun'],  # Bun
            row['Hemoglobin (g/dL)'],  # Hemoglobin
            row['Fibrinogen, Functional'],  # Fibrinogen
            row['gender'],  # Gender
            row['hyperlipidemia'],  # Hyperlipidemia
            row['obesity'],  # Obesity
            row['chronic_kidney_disease'],  # Chronic Kidney Disease
            row['Beta_blocker_used'],  # Beta Blocker Used
            row['warfarin_used'],  # Warfarin Used
            row['NOAC_used']   # NOAC Used
        ]
        prediction = predict_risk(features)
        predictions.append(prediction)
    
# 将预测结果添加到数据框中
    mdata['Prediction'] = predictions
    mdata.insert(0, 'id', range(1, len(mdata) + 1))
    # 返回带有预测结果的表格
    return render_template('cadpre.html', mdata=mdata.to_dict(orient='records'))


@app.route('/submit_data', methods=['POST'])
def submit_data():
    # 获取表单数据
    data = request.form

    # 定义分类变量的映射
    gender_map = {'0': 'Male', '1': 'Female'}
    hyperlipidemia_map = {'1': 'YES', '0': 'NO'}
    obesity_map = {'1': 'YES', '0': 'NO'}
    chronic_kidney_disease_map = {'1': 'YES', '0': 'NO'}
    beta_blocker_used_map = {'1': 'YES', '0': 'NO'}
    warfarin_used_map = {'1': 'YES', '0': 'NO'}
    noac_used_map = {'1': 'YES', '0': 'NO'}

    # 获取并打印每个字段值
    gcs = int(data.get('A'))  # GCS Score
    ckd_stage = data.get('B')  # CKD Stage
    height = float(data.get('C'))  # Height
    age = int(data.get('D'))  # Age
    platelet_count = float(data.get('E'))  # Platelet Count
    bun = float(data.get('F'))  # Bun
    hemoglobin = float(data.get('G'))  # Hemoglobin
    fibrinogen = float(data.get('H'))  # Fibrinogen, Functional
    gender = int(data.get('I'))  # Gender (Male = 0, Female = 1)
    hyperlipidemia = int(data.get('J'))  # Hyperlipidemia (YES = 1, NO = 0)
    obesity = int(data.get('K'))  # Obesity (YES = 1, NO = 0)
    chronic_kidney_disease = int(data.get('L'))  # Chronic Kidney Disease (YES = 1, NO = 0)
    beta_blocker_used = int(data.get('M'))  # Beta Blocker Used (YES = 1, NO = 0)
    warfarin_used = int(data.get('N'))  # Warfarin Used (YES = 1, NO = 0)
    noac_used = int(data.get('O'))  # NOAC Used (YES = 1, NO = 0)

    # 准备特征数据（按顺序）
    features = [
        gcs,
        ckd_stage,
        height, 
        age, 
        platelet_count, 
        bun, 
        hemoglobin, 
        fibrinogen, 
        gender, 
        hyperlipidemia, 
        obesity, 
        chronic_kidney_disease, 
        beta_blocker_used, 
        warfarin_used, 
        noac_used
    ]
    
    # 使用机器学习模型预测死亡风险
    prediction = predict_risk(features)


    # 返回预测结果给前端
    return render_template('cadpre.html', prediction=prediction,gcs=gcs, ckd_stage=ckd_stage, height=height, age=age, 
        platelet_count=platelet_count, bun=bun, hemoglobin=hemoglobin, fibrinogen=fibrinogen,
        gender=gender, hyperlipidemia=hyperlipidemia, obesity=obesity, 
        chronic_kidney_disease=chronic_kidney_disease, beta_blocker_used=beta_blocker_used,
        warfarin_used=warfarin_used, noac_used=noac_used,mdata=[])

if __name__ == '__main__':
    app.run(debug=False)

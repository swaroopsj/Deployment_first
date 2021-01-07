from flask import Flask, request, jsonify
from  server import util as util

app = Flask(__name__, static_url_path="/client", static_folder='../client', template_folder="../client")

@app.route('/', methods=['GET'])
def index():
    if request.method=="GET":
        return render_template("app.html")

@app.route('/get_gender')  ## use http://127.0.0.1:5000/hello
def get_gender():
    response = jsonify({
        'gender': util.get_gender()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_loan_status', methods=['POST'])
def estimated_status():
    gender = int(request.form['gender'])
    married = int(request.form['married'])
    dependents = int(request.form['dependents'])
    education = int(request.form['education'])
    self_employed = int(request.form['self_employed'])
    applicantincome = float(request.form['applicantincome'])
    coapplicantincome = float(request.form['coapplicantincome'])
    loanamount = float(request.form['loanamount'])
    loan_amount_term = float(request.form['loan_amount_term'])
    credit_history = float(request.form['credit_history'])
    property_area = int(request.form['property_area'])

    response = jsonify({
        'Loan_status': util.estimated_status(gender, married, dependents, education, self_employed, applicantincome,
                                             coapplicantincome, loanamount, loan_amount_term, credit_history,
                                             property_area)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    print("starting python flask server for loan status prediction")
    os.environ.get('PORT',5000)
    # util.load_saved_artifacts()
    app.run(port=port,debug=True,use_reload=False)

function getgenderValue() {
    var uigen = document.getElementsByName("uigen");
    for (var i in uigen) {
      if (uigen[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }
  
  function getmarriedValue() {
    var uimarried = document.getElementsByName("uimarried");
    for (var i in uimarried) {
      if (uimarried[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }

  function getdependentsValue() {
    var uidep = document.getElementsByName("uidep");
    for (var i in uidep) {
      if (uidep[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }

  function geteduValue() {
    var uiedu = document.getElementsByName("uiedu");
    for (var i in uiedu) {
      if (uiedu[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }

  function getempValue() {
    var uiemp = document.getElementsByName("uiemp");
    for (var i in uiemp) {
      if (uiemp[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }

  function getcreditValue() {
    var uicredit = document.getElementsByName("uicredit");
    for (var i in uicredit) {
      if (uicredit[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }

  function getareaValue() {
    var uiarea = document.getElementsByName("uiarea");
    for (var i in uiarea) {
      if (uiarea[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }


  function approvalstatuschecker() {
   console.log("approval status button clicked");
    var Gender = getgenderValue();
    var Married = getmarriedValue();
    var Dependents = getdependentsValue();
    var Education = geteduValue();
    var Self_Employed = getempValue();
    var ApplicantIncome = document.getElementById("uiappinc");
    var CoapplicantIncome = document.getElementById("uicoappinc");
    var LoanAmount = document.getElementById("uiloan");
    var Loan_Amount_Term = document.getElementById("uiloanterm");
    var Credit_History	= getcreditValue("uicredit");
    var Property_Area = getareaValue("uiarea");
    var approval_status =  document.getElementById("uiapprove");
    //approval_status.innerHTML ="<h2>" + "hello" + "</h2>";
    var url = "/predict_home_price"; 
     
    
    $.post(
      url,
      {
        gender: Gender,
        married: Married,
        dependents: Dependents,
        education: Education,
        self_employed:Self_Employed,
        applicantincome:ApplicantIncome.value,
        coapplicantincome:CoapplicantIncome.value,
        loanamount:LoanAmount.value,
        loan_amount_term:Loan_Amount_Term.value,
        credit_history:Credit_History,
        property_area:Property_Area
      },
      function (data, status) {
        console.log(data.Loan_status);
        approval_status.innerHTML ="<h2>" + data.Loan_status.toString() + "</h2>";
        console.log(status);
      }
    );
  }
  
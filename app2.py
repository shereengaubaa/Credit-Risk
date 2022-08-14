import pickle
import streamlit as st

pickle_in = open("pipeln.pkl", "rb")
pipeln = pickle.load(pickle_in)

def predict_loan_status(person_age, person_income, person_home_ownership, person_emp_length, loan_intent,loan_grade, loan_amnt, loan_int_rate, cb_person_default_on_file,cb_person_cred_hist_length  ):

    prediction = pipeln.predict([[person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate,cb_person_default_on_file, cb_person_cred_hist_length]])

    print(prediction)

    return prediction

def main():
    home_list= ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    options1 = list(range(len(home_list)))

    grade_list= ['G', 'F', 'E' , 'D', 'C', 'B', 'A']
    options3 = list(range(len(grade_list)))

    intent_list = ['EDUCATION', 'MEDICAL' ,'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']
    options2 = list(range(len(intent_list)))

    default_list = ['Y', 'N']
    options4 = list(range(len(default_list)))
    st.title("CREDIT RISK")
    html_temp = """
        <div style= "background-color:tomato;padding:10px">
        <h2 style = "color:white; text-align:center;">Check your loan approval status here!</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    person_age = st.slider("Enter your age" ,min_value=20, max_value=75, step=1)
    person_income = st.number_input("Enter your income" , min_value=4000, max_value=6000000)
    person_home_ownership = st.selectbox("Select your home ownership status",options1, format_func=lambda x: home_list[x] )
    person_emp_length= st.slider("Enter Employment length (in years)", min_value=0, max_value=45,step=1)
    loan_intent = st.selectbox("Why do you need the loan", options2, format_func=lambda x: intent_list[x])
    loan_grade = st.selectbox("Specify your loan grade", options3, format_func=lambda x: grade_list[x])
    loan_amnt = st.number_input("Specify your loan amount" , min_value=500, max_value=35000)
    loan_int_rate = st.number_input("Specify interest rate",min_value=5.42, max_value=23.22)
    cb_person_default_on_file = st.selectbox("Any historical default?" , options4, format_func=lambda x: default_list[x])
    cb_person_cred_hist_length = st.number_input("Credit history length?" , step=1)


    if st.button("Predict"):
        res = predict_loan_status(person_age, person_income, person_home_ownership, person_emp_length, loan_intent,loan_grade, loan_amnt, loan_int_rate, cb_person_default_on_file, cb_person_cred_hist_length )
        if res == 1:
            st.success('Loan status:  Approved')
        else:
            st.error('Loan status:  NOT Approved')







if __name__ == "__main__":
    main()
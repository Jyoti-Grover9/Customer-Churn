{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "52fa4d1f-f921-4b5c-bf00-8efdf04fc947",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load the dataset (adjust the path as per your file location)\n",
    "tel_churn = pd.read_csv('tel_churn.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "76c5675d-a6fc-45c1-9cb4-8205cc7482e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example: Data cleaning and preprocessing steps\n",
    "# Replace missing values, encode categorical variables, etc.\n",
    "tel_churn = tel_churn.dropna()  # Example of dropping rows with missing values\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "5dd337c8-4754-4690-b085-6999f7ba1c5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Example: Splitting into features (X) and target (y)\n",
    "X = tel_churn.drop('Churn', axis=1)  # Replace 'target_column_name' with your actual target column\n",
    "y = tel_churn['Churn']\n",
    "\n",
    "# Split data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "3d4b04bb-f708-4a58-9c28-a7649ee1a1e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "93085731-1a5e-4d96-a804-c88c1d66ab4e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7825159914712153"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf = RandomForestClassifier(n_estimators=150)\n",
    "\n",
    "clf.fit(X_train,y_train)\n",
    "\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "clf.score(X_test,y_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "17e459b8-6530-4181-a7d1-96c4df72bec4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Example: Initialize and train a RandomForestClassifier\n",
    "model = RandomForestClassifier()\n",
    "try:\n",
    "    model = RandomForestClassifier()\n",
    "    model.fit(X_train, y_train)\n",
    "except Exception as e:\n",
    "    print(f\"Error occurred: {e}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "19e5f8e8-28ca-4aa6-add4-1d19b307cafd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['tel_churn_model.pkl']"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(model, 'tel_churn_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5ce156c3-3705-4213-a007-5eab7457b197",
   "metadata": {},
   "outputs": [],
   "source": [
    "loaded_model = joblib.load('tel_churn_model.pkl')\n",
    "# Example: Predict using the loaded model\n",
    "predictions = loaded_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "b1b3fba5-db79-414a-a582-68bbe4fef0e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import streamlit as st \n",
    "from sklearn import preprocessing\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "aa3817e9-ae7b-4831-a4f3-8cc9912ac95f",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = pickle.load(open('tel_churn_model.pkl', 'rb'))\n",
    "cols=['customer_id','SeniorCitizen','MonthlyCharges','TotalCharges','Churn','gender_Female','gender_Male','Partner_No','Partner_Yes',\n",
    "      'PhoneService_Yes','MultipleLines_No',]    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "e5df72b8-17d5-45cb-b04c-491f56c316ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main(): \n",
    "    st.title(\"churn predictor\")\n",
    "    html_temp = \"\"\"\n",
    "    <div style=\"background:#025246 ;padding:10px\">\n",
    "    <h2 style=\"color:white;text-align:center;\">Income Prediction App </h2>\n",
    "    </div>\n",
    "    \"\"\"\n",
    "    st.markdown(html_temp, unsafe_allow_html = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "5603ae8f-8618-4f7d-acb5-632e6604483b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'Customer_id': 0, 'SeniorCitizen': 'Yes', 'MonthlyCharges': \"['0-1000']\", 'churn': '0', 'Gender': 'Female', 'Partner_No': '0', 'PhoneService': '0', 'MultipleLines': \"['0', '1']\"}\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "11 columns passed, passed data had 8 columns",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAssertionError\u001b[0m                            Traceback (most recent call last)",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\internals\\construction.py:939\u001b[0m, in \u001b[0;36m_finalize_columns_and_data\u001b[1;34m(content, columns, dtype)\u001b[0m\n\u001b[0;32m    938\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 939\u001b[0m     columns \u001b[38;5;241m=\u001b[39m _validate_or_indexify_columns(contents, columns)\n\u001b[0;32m    940\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mAssertionError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n\u001b[0;32m    941\u001b[0m     \u001b[38;5;66;03m# GH#26429 do not raise user-facing AssertionError\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\internals\\construction.py:986\u001b[0m, in \u001b[0;36m_validate_or_indexify_columns\u001b[1;34m(content, columns)\u001b[0m\n\u001b[0;32m    984\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m is_mi_list \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(columns) \u001b[38;5;241m!=\u001b[39m \u001b[38;5;28mlen\u001b[39m(content):  \u001b[38;5;66;03m# pragma: no cover\u001b[39;00m\n\u001b[0;32m    985\u001b[0m     \u001b[38;5;66;03m# caller's responsibility to check for this...\u001b[39;00m\n\u001b[1;32m--> 986\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mAssertionError\u001b[39;00m(\n\u001b[0;32m    987\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mlen\u001b[39m(columns)\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m columns passed, passed data had \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    988\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mlen\u001b[39m(content)\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m columns\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    989\u001b[0m     )\n\u001b[0;32m    990\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_mi_list:\n\u001b[0;32m    991\u001b[0m     \u001b[38;5;66;03m# check if nested list column, length of each sub-list should be equal\u001b[39;00m\n",
      "\u001b[1;31mAssertionError\u001b[0m: 11 columns passed, passed data had 8 columns",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[75], line 15\u001b[0m\n\u001b[0;32m     13\u001b[0m data \u001b[38;5;241m=\u001b[39m {\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mCustomer_id\u001b[39m\u001b[38;5;124m'\u001b[39m: \u001b[38;5;28mint\u001b[39m(Customer_id), \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mSeniorCitizen\u001b[39m\u001b[38;5;124m'\u001b[39m: SeniorCitizen, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mMonthlyCharges\u001b[39m\u001b[38;5;124m'\u001b[39m:MonthlyCharges, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mchurn\u001b[39m\u001b[38;5;124m'\u001b[39m: churn, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mGender\u001b[39m\u001b[38;5;124m'\u001b[39m: gender, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPartner_No\u001b[39m\u001b[38;5;124m'\u001b[39m: Partner_No, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPhoneService\u001b[39m\u001b[38;5;124m'\u001b[39m: PhoneService, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mMultipleLines\u001b[39m\u001b[38;5;124m'\u001b[39m: MultipleLines}\n\u001b[0;32m     14\u001b[0m \u001b[38;5;28mprint\u001b[39m(data)\n\u001b[1;32m---> 15\u001b[0m df\u001b[38;5;241m=\u001b[39mpd\u001b[38;5;241m.\u001b[39mDataFrame([\u001b[38;5;28mlist\u001b[39m(data\u001b[38;5;241m.\u001b[39mvalues())], columns\u001b[38;5;241m=\u001b[39m[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcustomer_id\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mSeniorCitizen\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mMonthlyCharges\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mTotalCharges\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mChurn\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgender_Female\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgender_Male\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPartner_No\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPartner_Yes\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPhoneService_Yes\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mMultipleLines_No\u001b[39m\u001b[38;5;124m'\u001b[39m]    \n\u001b[0;32m     16\u001b[0m )\n\u001b[0;32m     17\u001b[0m category_col \u001b[38;5;241m=\u001b[39m[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcustomer_id\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mSeniorCitizen\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mMonthlyCharges\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mTotalCharges\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mChurn\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgender_Female\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgender_Male\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPartner_No\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPartner_Yes\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPhoneService_Yes\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mMultipleLines_No\u001b[39m\u001b[38;5;124m'\u001b[39m]    \n\u001b[0;32m     19\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m cat \u001b[38;5;129;01min\u001b[39;00m encoder_dict:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\frame.py:806\u001b[0m, in \u001b[0;36mDataFrame.__init__\u001b[1;34m(self, data, index, columns, dtype, copy)\u001b[0m\n\u001b[0;32m    804\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m columns \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m    805\u001b[0m         columns \u001b[38;5;241m=\u001b[39m ensure_index(columns)\n\u001b[1;32m--> 806\u001b[0m     arrays, columns, index \u001b[38;5;241m=\u001b[39m nested_data_to_arrays(\n\u001b[0;32m    807\u001b[0m         \u001b[38;5;66;03m# error: Argument 3 to \"nested_data_to_arrays\" has incompatible\u001b[39;00m\n\u001b[0;32m    808\u001b[0m         \u001b[38;5;66;03m# type \"Optional[Collection[Any]]\"; expected \"Optional[Index]\"\u001b[39;00m\n\u001b[0;32m    809\u001b[0m         data,\n\u001b[0;32m    810\u001b[0m         columns,\n\u001b[0;32m    811\u001b[0m         index,  \u001b[38;5;66;03m# type: ignore[arg-type]\u001b[39;00m\n\u001b[0;32m    812\u001b[0m         dtype,\n\u001b[0;32m    813\u001b[0m     )\n\u001b[0;32m    814\u001b[0m     mgr \u001b[38;5;241m=\u001b[39m arrays_to_mgr(\n\u001b[0;32m    815\u001b[0m         arrays,\n\u001b[0;32m    816\u001b[0m         columns,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    819\u001b[0m         typ\u001b[38;5;241m=\u001b[39mmanager,\n\u001b[0;32m    820\u001b[0m     )\n\u001b[0;32m    821\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\internals\\construction.py:520\u001b[0m, in \u001b[0;36mnested_data_to_arrays\u001b[1;34m(data, columns, index, dtype)\u001b[0m\n\u001b[0;32m    517\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_named_tuple(data[\u001b[38;5;241m0\u001b[39m]) \u001b[38;5;129;01mand\u001b[39;00m columns \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m    518\u001b[0m     columns \u001b[38;5;241m=\u001b[39m ensure_index(data[\u001b[38;5;241m0\u001b[39m]\u001b[38;5;241m.\u001b[39m_fields)\n\u001b[1;32m--> 520\u001b[0m arrays, columns \u001b[38;5;241m=\u001b[39m to_arrays(data, columns, dtype\u001b[38;5;241m=\u001b[39mdtype)\n\u001b[0;32m    521\u001b[0m columns \u001b[38;5;241m=\u001b[39m ensure_index(columns)\n\u001b[0;32m    523\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m index \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\internals\\construction.py:845\u001b[0m, in \u001b[0;36mto_arrays\u001b[1;34m(data, columns, dtype)\u001b[0m\n\u001b[0;32m    842\u001b[0m     data \u001b[38;5;241m=\u001b[39m [\u001b[38;5;28mtuple\u001b[39m(x) \u001b[38;5;28;01mfor\u001b[39;00m x \u001b[38;5;129;01min\u001b[39;00m data]\n\u001b[0;32m    843\u001b[0m     arr \u001b[38;5;241m=\u001b[39m _list_to_arrays(data)\n\u001b[1;32m--> 845\u001b[0m content, columns \u001b[38;5;241m=\u001b[39m _finalize_columns_and_data(arr, columns, dtype)\n\u001b[0;32m    846\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m content, columns\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\internals\\construction.py:942\u001b[0m, in \u001b[0;36m_finalize_columns_and_data\u001b[1;34m(content, columns, dtype)\u001b[0m\n\u001b[0;32m    939\u001b[0m     columns \u001b[38;5;241m=\u001b[39m _validate_or_indexify_columns(contents, columns)\n\u001b[0;32m    940\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mAssertionError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n\u001b[0;32m    941\u001b[0m     \u001b[38;5;66;03m# GH#26429 do not raise user-facing AssertionError\u001b[39;00m\n\u001b[1;32m--> 942\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(err) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[0;32m    944\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(contents) \u001b[38;5;129;01mand\u001b[39;00m contents[\u001b[38;5;241m0\u001b[39m]\u001b[38;5;241m.\u001b[39mdtype \u001b[38;5;241m==\u001b[39m np\u001b[38;5;241m.\u001b[39mobject_:\n\u001b[0;32m    945\u001b[0m     contents \u001b[38;5;241m=\u001b[39m convert_object_array(contents, dtype\u001b[38;5;241m=\u001b[39mdtype)\n",
      "\u001b[1;31mValueError\u001b[0m: 11 columns passed, passed data had 8 columns"
     ]
    }
   ],
   "source": [
    "ge = st.text_input(\"Customer_id\",\"0\") \n",
    "Customer_id= st.text_input(\"Customer_id\",\"0\") \n",
    "SeniorCitizen = st.selectbox(\"SeniorCitizen\",[\"Yes\",\"No\"]) \n",
    "MonthlyCharges = st.text_input(\"MonthlyCharges\",[\"0-1000\"])\n",
    "TotalCharges = st.text_input(\"TotalCharges\",\"0\") \n",
    "churn = st.selectbox(\"churn\",[\"0\",\"1\"]) \n",
    "gender = st.selectbox(\"Gender\",[\"Female\",\"Male\"]) \n",
    "Partner_No = st.selectbox(\"Partner\",[\"0\",\"1\"]) \n",
    "PhoneService = st.selectbox(\"PhoneService\",[\"0\",\"1\"]) \n",
    "MultipleLines= st.text_input(\"MultipleLines\",[\"0\",\"1\"]) \n",
    "if st.button(\"Predict\"): \n",
    "      features = ['Customer_id','SeniorCitizen','MonthlyCharges','TotalCharges','Churn','gender_Female','gender_Male','Partner_No','Partner_Yes','PhoneService_Yes','MultipleLines_No']    \n",
    "data = {'Customer_id': int(Customer_id), 'SeniorCitizen': SeniorCitizen, 'MonthlyCharges':MonthlyCharges, 'churn': churn, 'Gender': gender, 'Partner_No': Partner_No, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines}\n",
    "print(data)\n",
    "df=pd.DataFrame([list(data.values())], columns=['customer_id','SeniorCitizen','MonthlyCharges','TotalCharges','Churn','gender_Female','gender_Male','Partner_No','Partner_Yes','PhoneService_Yes','MultipleLines_No']    \n",
    ")\n",
    "category_col =['customer_id','SeniorCitizen','MonthlyCharges','TotalCharges','Churn','gender_Female','gender_Male','Partner_No','Partner_Yes','PhoneService_Yes','MultipleLines_No']    \n",
    "\n",
    "for cat in encoder_dict:\n",
    "            for col in df.columns:\n",
    "                le = preprocessing.LabelEncoder()\n",
    "                if cat == col:\n",
    "                    le.classes_ = encoder_dict[cat]\n",
    "                    for unique_item in df[col].unique():\n",
    "                        if unique_item not in le.classes_:\n",
    "                            df[col] = ['Unknown' if x == unique_item else x for x in df[col]]\n",
    "                    df[col] = le.transform(df[col])\n",
    "            \n",
    "                    features_list = df.values.tolist()      \n",
    "                    prediction = model.predict(features_list)\n",
    "    \n",
    "                    output = int(prediction[0])\n",
    "                    if output == 1:\n",
    "                         text = \"Leave\"\n",
    "                    else:\n",
    "                         text = \"Not Leave\"\n",
    "\n",
    "                    st.success('The custmer will {}'.format(text))\n",
    "      \n",
    "if __name__=='__main__': \n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count":"20000" ,
   "id": "ab1958cc-4e2b-4be5-b8e7-fb04af4fab5e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

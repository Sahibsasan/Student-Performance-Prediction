
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("🎓 Student Performance Prediction")

data={
"Attendance":[92,85,74,60,95,45,80,70,88,55,98,67,76,82,40,90,58,72,86,50,91,64,79,83,47,96,69,73,89,52],
"Internal_Marks":[28,24,18,15,29,10,22,17,25,13,30,16,19,23,8,27,14,18,26,12,28,15,21,24,9,29,17,19,25,11],
"Assignment_Marks":[18,16,12,10,19,7,15,11,17,9,20,10,13,16,6,18,9,12,17,8,18,10,14,16,7,19,11,13,17,8],
"Study_Hours":[5,4,3,2,6,1,4,3,5,2,6,3,3,4,1,5,2,3,5,2,5,2,4,4,1,6,3,3,5,2],
"Previous_Score":[82,75,60,48,90,40,70,58,80,50,95,55,65,74,35,85,52,61,79,45,84,56,68,76,42,92,59,63,81,49],
"Result":["Pass","Pass","Fail","Fail","Pass","Fail","Pass","Fail","Pass","Fail","Pass","Fail","Fail","Pass","Fail","Pass","Fail","Fail","Pass","Fail","Pass","Fail","Pass","Pass","Fail","Pass","Fail","Fail","Pass","Fail"]}
df=pd.DataFrame(data)
df["Result_Code"]=df["Result"].map({"Fail":0,"Pass":1})
X=df[["Attendance","Internal_Marks","Assignment_Marks","Study_Hours","Previous_Score"]]
y=df["Result_Code"]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
model=RandomForestClassifier(random_state=42)
model.fit(X_scaled,y)

tab1,tab2,tab3=st.tabs(["Dataset","Prediction","Feature Importance"])
with tab1:
 st.dataframe(df)
 st.write(df.describe())
 fig,ax=plt.subplots()
 ax.scatter(df["Attendance"],df["Previous_Score"],c=y)
 ax.set_xlabel("Attendance");ax.set_ylabel("Previous Score")
 st.pyplot(fig)
with tab2:
 a=st.slider("Attendance",0,100,80)
 i=st.slider("Internal Marks",0,30,20)
 am=st.slider("Assignment Marks",0,20,15)
 sh=st.slider("Study Hours",0,10,4)
 ps=st.slider("Previous Score",0,100,65)
 if st.button("Predict"):
  inp=scaler.transform([[a,i,am,sh,ps]])
  pred=model.predict(inp)[0]
  prob=model.predict_proba(inp).max()*100
  if pred==1: st.success(f"Predicted Result: PASS ({prob:.1f}% confidence)")
  else: st.error(f"Predicted Result: FAIL ({prob:.1f}% confidence)")
with tab3:
 imp=pd.DataFrame({"Feature":X.columns,"Importance":model.feature_importances_}).sort_values("Importance")
 fig,ax=plt.subplots()
 ax.barh(imp["Feature"],imp["Importance"])
 st.pyplot(fig)

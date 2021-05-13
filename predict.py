import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import base64

st.write("""
# Simple Fake News Prediction App

This app predicts the **News** type!
""")

st.header('Input Fake News Data')
df=pd.read_csv("data/dfcomplete.csv")
def Page1(page):
    t = {
    "ABC News Politics": 1,
    "Addicting Info": 2,
    "CNN Politics": 3,
    "Eagle rising": 4,
    "Freedom Daily": 5,
    "Occupy Democrats": 6,
    "Politico": 7,
    "Right Wing News": 8,
    "The Other 98%": 9
    }
    return t[page]

def Category1(category):
    t = {
    "left": 1,
    "mainstream": 2,
    "right": 3
    }
    return t[category]


def Posttype1(post_type):
    t = {
    "link": 1,
    "photo": 2,
    "text": 3,
    "video": 4
    }
    return t[post_type]


def debate1(debate):
    t = {
    "Yes": 1,
    "No": 0,
    }
    return t[debate]




def user_input_features():
    l1=["left","mainstream","right"]
    l2=["ABC News Politics","Addicting Info","CNN Politics","Eagle rising","Freedom Daily","Occupy Democrats","Politico","Right Wing News","The Other 98%"]
    l3=["link","photo","text","video"]
    l4=["Yes","No"]
    Category = st.selectbox('Select the Category', l1,index=0)
    Page = st.selectbox('Select the Page', l2,index=0)
    Post_Type = st.selectbox('Select the Post type', l3,index=0)
    Debate= st.selectbox('Choose if debate happened or not', l4,index=0)
    share_count=st.slider('Share Count', 1, 1088995, 342925)
    reaction_count=st.slider('Reaction Count', 2, 456458,199325)
    comment_count=st.slider('Comment Count', 1, 159047,86086)
    Day=st.slider('Day', 19, 27, 21)
    data = {'Category': Category,
            'Page': Page,
            'Post Type': Post_Type,
            'Debate': Debate,
            'share_count':share_count,
            'reaction_count':reaction_count,
            'comment_count':comment_count,
            'Day':Day}
    data1 = {'Category': Category1(Category),
            'Page': Page1(Page),
            'Post Type': Posttype1(Post_Type),
            'Debate': debate1(Debate),
            'share_count':share_count,
            'reaction_count':reaction_count,
            'comment_count':comment_count,
            'Day':Day}
    st.subheader('User Input parameters')
    features1 = pd.DataFrame(data, index=[0])
    features = pd.DataFrame(data1, index=[0])
    st.write(features1)
    return features
df1 = user_input_features()
if st.button("Make predictions"):
    c="Rating"
    X=df.loc[:,df.columns != c]
    Y=df[c]
    clf = RandomForestClassifier()
    clf.fit(X, Y)
    st.subheader('Prediction')
    pred=clf.predict(df1)
    def rating(n):
        t = {
            4: 'no factual content',
            3: 'mostly true',
            1: 'mixture of true and false',
            2: 'mostly false'
            }
        return (t[n])
    st.write("The news type is: ",rating(pred[0]))
    st.subheader('Prediction Probability')
    proba = clf.predict_proba(df1)
    t = {
        'mixture of true and false': proba[0][0],
        'mostly false': proba[0][1],
        'mostly true': proba[0][2],
        'no factual content': proba[0][3]
        }
    f=pd.DataFrame(t,index=[0])
    st.write(f)
    df1["Rating"]=pred[0]
    df=df.append(df1, ignore_index=True)
    df.to_csv("data/dfcomplete.csv",index=False)

    def filedownload(df, filename):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
            return href
    st.markdown(filedownload(df,'Test_data.csv'), unsafe_allow_html=True)

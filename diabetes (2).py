import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tkinter
# import all required machine learning packages
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# clf=classification()
# clf.ploting()
# clf.calling()
window = tkinter.Tk()
window.geometry("800x800")
window.title("DIABETES PREDICTION SOFTWARE")
label = tkinter.Label(text="Diabetes Prediction System",
                      font=("arial", 30), fg='green', bg='white')
label.place(x=100, y=10)
label1 = tkinter.Label(text="Pregnancies", font=("arial", 15), borderwidth=0)
label2 = tkinter.Label(text="Glucose", font=("arial", 15), borderwidth=0)
label3 = tkinter.Label(text="BloodPressure", font=("arial", 15), borderwidth=0)
label4 = tkinter.Label(text="SkinThickness", font=("arial", 15), borderwidth=0)
label5 = tkinter.Label(text="Insulin", font=("arial", 15), borderwidth=0)
label6 = tkinter.Label(text="BMI", font=("arial", 15), borderwidth=0)
label7 = tkinter.Label(text="DiabetesPedigreeFunction",
                       font=("arial", 15), borderwidth=0)
label8 = tkinter.Label(text="Age", font=("arial", 15), borderwidth=0)
label1.place(x=50, y=100)
label2.place(x=50, y=150)
label3.place(x=50, y=200)
label4.place(x=50, y=250)
label5.place(x=50, y=300)
label6.place(x=50, y=350)
label7.place(x=50, y=400)
label8.place(x=50, y=450)
text1 = tkinter.IntVar()
text2 = tkinter.IntVar()
text3 = tkinter.IntVar()
text4 = tkinter.IntVar()
text5 = tkinter.IntVar()
text6 = tkinter.DoubleVar()
text7 = tkinter.DoubleVar()
text8 = tkinter.IntVar()
entry1 = tkinter.Entry(textvariable=text1, font=("arial", 15), borderwidth=0)
entry1.place(x=300, y=100)
entry2 = tkinter.Entry(textvariable=text2, font=("arial", 15), borderwidth=0)
entry2.place(x=300, y=150)
entry3 = tkinter.Entry(textvariable=text3, font=("arial", 15), borderwidth=0)
entry3.place(x=300, y=200)
entry4 = tkinter.Entry(textvariable=text4, font=("arial", 15), borderwidth=0)
entry4.place(x=300, y=250)
entry5 = tkinter.Entry(textvariable=text5, font=("arial", 15), borderwidth=0)
entry5.place(x=300, y=300)
entry6 = tkinter.Entry(textvariable=text6, font=("arial", 15), borderwidth=0)
entry6.place(x=300, y=350)
entry7 = tkinter.Entry(textvariable=text7, font=("arial", 15), borderwidth=0)
entry7.place(x=300, y=400)
entry8 = tkinter.Entry(textvariable=text8, font=("arial", 15), borderwidth=0)
entry8.place(x=300, y=450)


class classification:
    def __init__(self):
        self.data = pd.read_csv("diabetes.csv")
        data = self.data
        print("The datasets is \n", self.data)
        print("The null values of the dataset is \n", data.isnull().sum())
        print("The description of the numerical data is \n", data.describe())
        print("The information about the data is\n", data.info())
        print("The number of duplicates present in the dataset are\n",
              data.duplicated().sum())
        print(data.columns)
        self.labels = self.data.columns
        print(self.labels)
        self.label1 = tkinter.Label()
        self.label2 = tkinter.Label()
        self.label3 = tkinter.Label()

    def ploting(self):
        sns.countplot(x="Outcome", data=self.data, label="Yes")
        plt.legend()
        plt.title("OUTCOMES")
        plt.show()
        print(self.data)
        labels = self.data.columns
        labels = list(labels)
        labels.remove("Outcome")
        for label in labels:
            plt.title(label)
            plt.hist(self.data[self.data["Outcome"] == 0][label],
                     label="Yes", color="blue", alpha=0.5, density=True)
            plt.hist(self.data[self.data["Outcome"] == 1][label],
                     label="No", color="red", alpha=0.5, density=True)
            plt.xlabel(label)
            plt.ylabel("Outcome")
            plt.legend()
            plt.show()
        # sns.pairplot(self.data,hue="Outcome")
        # plt.title("PAIRPLOTS")
        # plt.show()

    def spliting(self):
        self.X_features = list(self.data.columns)
        self.X_features.remove("Outcome")
        self.X_data = self.data[self.X_features].values
        self.y_data = self.data["Outcome"].values
        print("The independent variabels are ", self.X_data)
        print("The dependent variables area ", self.y_data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_data, self.y_data, random_state=101, train_size=0.75)

    def classifires(self):
        self.logreg = LogisticRegression(max_iter=len(self.X_train))
        self.logreg.fit(self.X_train, self.y_train)
        self.tree = DecisionTreeClassifier(max_depth=10)
        self.tree.fit(self.X_train, self.y_train)
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(self.X_train, self.y_train)
        self.svm = SVC()
        self.svm.fit(self.X_train, self.y_train)

    def prediction(self):
        self.y_pred1 = self.logreg.predict(self.X_test)
        self.y_pred2 = self.tree.predict(self.X_test)
        self.y_pred3 = self.knn.predict(self.X_test)
        self.y_pred4 = self.svm.predict(self.X_test)

    def classf(self):
        report = classification_report(self.y_pred1, self.y_test)
        print("The classification report for logistic regression is \n", report)
        report = classification_report(self.y_pred2, self.y_test)
        print("The classification report for DTC is \n", report)
        report = classification_report(self.y_pred3, self.y_test)
        print("The classification report for KNN is \n", report)
        report = classification_report(self.y_pred4, self.y_test)
        print("The classification report for SVM is \n", report)

    def conf(self):
        self.conf1 = confusion_matrix(self.y_test, self.y_pred1)
        self.conf2 = confusion_matrix(self.y_test, self.y_pred2)
        self.conf3 = confusion_matrix(self.y_test, self.y_pred3)
        self.conf4 = confusion_matrix(self.y_test, self.y_pred4)
        sns.heatmap(self.conf1, annot=True)
        plt.title("LOGISTIC REGRESSION")
        plt.show()
        sns.heatmap(self.conf2, annot=True)
        plt.title("DTC")
        plt.show()
        sns.heatmap(self.conf3, annot=True)
        plt.title("KNN")
        plt.show()
        sns.heatmap(self.conf4, annot=True)
        plt.title("SVM")
        plt.show()

    def accur(self):
        self.acc1 = accuracy_score(self.y_pred1, self.y_test)
        self.acc2 = accuracy_score(self.y_pred2, self.y_test)
        self.acc3 = accuracy_score(self.y_pred3, self.y_test)
        self.acc4 = accuracy_score(self.y_pred4, self.y_test)
        scores = [self.acc1, self.acc2, self.acc3, self.acc4]
        names = ["LR", "DTC", "KNN", "SVM"]
        sns.barplot(x=names, y=scores, label=names)
        plt.xlabel("Classifires")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

    def predict_b(self):
        var1 = text1.get()
        var2 = text2.get()
        var3 = text3.get()
        var4 = text4.get()
        var5 = text5.get()
        var7 = text6.get()
        var6 = text7.get()
        var8 = text8.get()
        x = np.array([[var1, var2, var3, var4, var5, var6, var7, var8]])
        y_pred = self.svm.predict(x)
        print(y_pred)
        x = np.array([[var1, var2, var3, var4, var5, var6, var7, var8]])
        y_pred = self.svm.predict(x)
        if y_pred == 1:
            self.label1 = tkinter.Label(
                text="Yes", font=("arial", 15), borderwidth=0)
            self.label1.place(x=800, y=600)
        else:
            self.label2 = tkinter.Label(
                text="No", font=("arial", 15), borderwidth=0)
            self.label2.place(x=800, y=600)

    def reset(self):
        entry1.delete(0, tkinter.END)
        entry2.delete(0, tkinter.END)
        entry3.delete(0, tkinter.END)
        entry4.delete(0, tkinter.END)
        entry5.delete(0, tkinter.END)
        entry6 .delete(0, tkinter.END)
        entry7.delete(0, tkinter.END)
        entry8 .delete(0, tkinter.END)

    def clear(self):
        self.label1.after(100, self.label1.destroy())
        self.label2.after(100, self.label2.destroy())

    def calling(self):
        self.spliting()
        self.classifires()
        self.prediction()
        self.classf()
        self.conf()
        self.accur()


clf = classification()
clf.ploting()
clf.calling()


button1 = tkinter.Button(text="submit", borderwidth=0,
                         font=("arial", 15), command=clf.predict_b)
button1.place(x=200, y=600)
button2 = tkinter.Button(text="reset", borderwidth=0,
                         font=("arial", 15), command=clf.reset)
button2.place(x=400, y=600)
button3 = tkinter.Button(text="clear", borderwidth=0,
                         font=("arial", 15), command=clf.clear)
button3.place(x=600, y=600)
window.mainloop()

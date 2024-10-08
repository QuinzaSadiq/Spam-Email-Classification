import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from io import BytesIO
import base64

from django.shortcuts import render

# Helper function to save plot as base64
def get_plot():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

def index(request):
    # Load dataset
    data = pd.read_csv('emails.csv')

    # Preprocessing: Dropping unnecessary columns and handling missing values if any
    data.drop('Email No.', axis=1, inplace=True)
    
    # Get the head of the dataset to display
    dataset_head = data.head().to_html(classes='table table-striped', index=False)

    # Get shape information of the dataset
    dataset_shape = data.shape

    # Get describe information of the dataset
    dataset_description = data.describe().to_html(classes='table table-bordered', index=True)

    # Splitting data into train and test
    x = data.drop('Prediction', axis=1)
    y = data['Prediction']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train and evaluate Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(x_train, y_train)
    y_pred_logistic = logistic_model.predict(x_test)
    logistic_accuracy = accuracy_score(y_test, y_pred_logistic) * 100
    logistic_precision = precision_score(y_test, y_pred_logistic, average='binary') * 100
    logistic_recall = recall_score(y_test, y_pred_logistic, average='binary') * 100
    logistic_f1 = f1_score(y_test, y_pred_logistic, average='binary') * 100

    # Train and evaluate Naive Bayes
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(x_train, y_train)
    y_pred_naive_bayes = naive_bayes_model.predict(x_test)
    naive_bayes_accuracy = accuracy_score(y_test, y_pred_naive_bayes) * 100
    naive_bayes_precision = precision_score(y_test, y_pred_naive_bayes, average='binary') * 100
    naive_bayes_recall = recall_score(y_test, y_pred_naive_bayes, average='binary') * 100
    naive_bayes_f1 = f1_score(y_test, y_pred_naive_bayes, average='binary') * 100

    # Train and evaluate K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(x_train, y_train)
    y_pred_knn = knn_model.predict(x_test)
    knn_accuracy = accuracy_score(y_test, y_pred_knn) * 100
    knn_precision = precision_score(y_test, y_pred_knn, average='binary') * 100
    knn_recall = recall_score(y_test, y_pred_knn, average='binary') * 100
    knn_f1 = f1_score(y_test, y_pred_knn, average='binary') * 100

    # Train and evaluate Decision Tree
    decision_tree_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
    decision_tree_model.fit(x_train, y_train)
    y_pred_decision_tree = decision_tree_model.predict(x_test)
    decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree) * 100
    decision_tree_precision = precision_score(y_test, y_pred_decision_tree, average='binary') * 100
    decision_tree_recall = recall_score(y_test, y_pred_decision_tree, average='binary') * 100
    decision_tree_f1 = f1_score(y_test, y_pred_decision_tree, average='binary') * 100

    # Prepare accuracy data for the pie chart
    accuracies = [logistic_accuracy, naive_bayes_accuracy, knn_accuracy, decision_tree_accuracy]
    model_names = ["Logistic Regression", "Naive Bayes", "K-Nearest Neighbors", "Decision Tree"]

    # Plotting the accuracy pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(accuracies, labels=model_names, autopct='%1.1f%%', startangle=140, colors=['pink', 'skyblue', 'teal', 'salmon'])
    plt.title("Accuracy Distribution of Four Models")
    plt.axis('equal')

    # Convert the plot to base64 string to render it in HTML
    plot_url = get_plot()

    # Render the template and pass the context data with accuracy, precision, recall, and F1 scores
    context = {
        'dataset_head': dataset_head,
        'dataset_shape': dataset_shape,
        'dataset_description': dataset_description,
        'logistic_accuracy': logistic_accuracy,
        'logistic_precision': logistic_precision,
        'logistic_recall': logistic_recall,
        'logistic_f1': logistic_f1,
        'naive_bayes_accuracy': naive_bayes_accuracy,
        'naive_bayes_precision': naive_bayes_precision,
        'naive_bayes_recall': naive_bayes_recall,
        'naive_bayes_f1': naive_bayes_f1,
        'knn_accuracy': knn_accuracy,
        'knn_precision': knn_precision,
        'knn_recall': knn_recall,
        'knn_f1': knn_f1,
        'decision_tree_accuracy': decision_tree_accuracy,
        'decision_tree_precision': decision_tree_precision,
        'decision_tree_recall': decision_tree_recall,
        'decision_tree_f1': decision_tree_f1,
        'plot_url': plot_url,
    }
    
    return render(request, 'classifier/index.html', context)

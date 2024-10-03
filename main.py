from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import difflib

app = Flask(__name__)
os.system("start \"\" http://127.0.0.1:5000")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        def preprocess_text(text):
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in stop_words]
            return ' '.join(words)

        symptoms = pd.read_csv('ayurvedic_symptoms_desc_updated.csv')
        data = pd.read_csv('Symptom2Disease.csv')
        data.drop(columns=["Unnamed: 0"], inplace=True)
        df1 = pd.read_csv('Formulation-Indications.csv')

        labels = data['label']
        symptoms = data['text']

        stop_words = set(stopwords.words('english'))

        preprocessed_symptoms = symptoms.apply(preprocess_text)

        tfidf_vectorizer = TfidfVectorizer(max_features=2000)
        tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_symptoms).toarray()

        X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)

        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(X_train, y_train)
        predictions = knn_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        # print(f'Accuracy: {accuracy:.2f}')
        classificationreport = classification_report(y_test, predictions, output_dict=True)

        # Convert the classification report dictionary to a Pandas DataFrame
        df = pd.DataFrame(classificationreport).transpose()

        # Convert DataFrame to HTML table
        classificationreport = df.to_html()

        symptom = request.form.get('symptomInput')
        preprocessed_symptom = preprocess_text(symptom)
        symptom_tfidf = tfidf_vectorizer.transform([preprocessed_symptom])
        predicted_disease = knn_classifier.predict(symptom_tfidf)
        pred_disease = f'Predicted Disease: {predicted_disease[0]}'

        data1 = pd.read_csv('ayurvedic_symptoms_desc_updated.csv')
        # print(data1)
        words = symptom.split(", " or " ")
        data1['common_words'] = data1['English_Symptoms'].apply(lambda x: sum(word.lower() in x.lower() for word in words))
        filtered_data = data1[data1['common_words'] > 0]
        filtered_data = filtered_data.sort_values(by='common_words', ascending=False)
        filtered_data = filtered_data.drop(columns=['common_words'])
        original_data_same_indices = data1.loc[filtered_data.index]

        original_data_same_indices = original_data_same_indices.head(10)

        formulations_lst = list(df1['Name of Medicine'])
        original_list = list(df1['Main Indications'])

        processed_list = []
        for item in original_list:
            processed_item = ''.join(item.split()).lower()
            processed_list.append(processed_item)

        list_of_symptoms = processed_list
        flat_symptoms = [symptom.replace(',', ' ').split() for symptoms in list_of_symptoms for symptom in symptoms.split(',')]
        unique_symptoms = list(set(symptom for sublist in flat_symptoms for symptom in sublist))

        symptoms = pd.read_csv('ayurvedic_symptoms_desc_updated.csv')

        symptoms['Symptom'] = symptoms['Symptom'].str.lower()

        correct_words = unique_symptoms
        data2 = {
            "Formulation": formulations_lst,
            "Symptoms": processed_list,
        }
        df = pd.DataFrame(data2)

        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(df['Symptoms'])

        clf = MultinomialNB()
        clf.fit(X_tfidf, df['Formulation'])
        def get_column_values(df, column_name):
            # Get the column values as a list
            column_values = df[column_name].tolist()

            # Convert the list to a string with space separation
            column_values_str = ' '.join(map(str, column_values))

            return column_values_str

        user_input = get_column_values(original_data_same_indices, 'Symptom')
        print("*"*100,"\n",original_data_same_indices)
        def symptoms_desc(symptom_name):
            row = symptoms[symptoms['Symptom'] == symptom_name.lower()]
            if not row.empty:
                description = row.iloc[0]['Description']
                print(f'Description of "{symptom_name}": {description}')
            else:
                print(f'Symptom "{symptom_name}" not found in the DataFrame.')

        def symptoms_lst_desc(user_symptoms):
            for item in user_symptoms:
                symptoms_desc(item)

        def correct_symptoms(symptoms):
            corrected_symptoms = []
            for symptom in symptoms:
                corrected_symptom = difflib.get_close_matches(symptom, correct_words, n=1, cutoff=0.5)
                if corrected_symptom:
                    corrected_symptoms.append(corrected_symptom[0])
                else:
                    corrected_symptoms.append(symptom)
            return corrected_symptoms

        input_symptoms = user_input.split()
        # print("*"*100,"\n",input_symptoms)
        new_symptoms = correct_symptoms(input_symptoms)

        symptoms_lst_desc(new_symptoms)

        new_symptoms_tfidf = tfidf_vectorizer.transform(new_symptoms)
        predicted_label = clf.predict(new_symptoms_tfidf)

        c = len(original_data_same_indices) if len(original_data_same_indices) < 10 else 10
        medicines = []
        while (c > 0):
            meds = []
            mask = df1.iloc[:, 0].isin([predicted_label[len(original_data_same_indices) - c]])
            filtered_df = df1[mask]
            for index, row in filtered_df.iterrows():
                meds.append(str(row))
            c -= 1
            medicines.append(meds)
        return render_template('index.html',
                               predictedDisease=pred_disease,
                               prescribedMedicines=medicines[0:3],
                               accuracy = f'Accuracy: {accuracy:.2f}',
                               classificationreport = classificationreport)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

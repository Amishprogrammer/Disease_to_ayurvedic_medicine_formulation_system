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

def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())
    # Removing stopwords and non-alphabetic characters
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

def get_column_values(df, column_name):
    # Get the column values as a list
    column_values = df[column_name].tolist()

    # Convert the list to a string with space separation
    column_values_str = ' '.join(map(str, column_values))

    return column_values_str

def symptoms_desc(symptom_name):
    row = symptoms[symptoms['Symptom'] == symptom_name.lower()]
#     print(row)
    if not row.empty:
        description = row.iloc[0]['Description']
        print(f'Description of "{symptom_name}": {description}')
    else:
        print(f'Symptom "{symptom_name}" not found in the DataFrame.')

def symptoms_lst_desc(user_symptoms):
    for item in user_symptoms:
#         print(item)
        symptoms_desc(item)

def correct_symptoms(symptoms):
    corrected_symptoms = []
    for symptom in symptoms:
        corrected_symptom = difflib.get_close_matches(symptom, correct_words, n=1, cutoff=0.6)
        if corrected_symptom:
            corrected_symptoms.append(corrected_symptom[0])
        else:
            corrected_symptoms.append(symptom)
    return corrected_symptoms

    
app = Flask(__name__)

symptoms = pd.read_csv('ayurvedic_symptoms_desc_updated.csv')
data = pd.read_csv('Symptom2Disease.csv')
data.drop(columns=["Unnamed: 0"], inplace=True)
data1 = pd.read_csv('ayurvedic_symptoms_desc_updated.csv')
df1 = pd.read_csv('Formulation-Indications.csv')
labels = data['label']  # Contains the labels or categories associated with the text data
symptoms = data['text']  # Contains the textual data (e.g., symptoms, sentences) for analysis

stop_words = set(stopwords.words('english'))

# Apply preprocessing to symptoms

preprocessed_symptoms = symptoms.apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer(max_features=2000)  # You can adjust max_features based on your dataset size
tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_symptoms).toarray()

X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)

# KNN Model Training

knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) based on your dataset
knn_classifier.fit(X_train, y_train)

# Predictions

predictions = knn_classifier.predict(X_test)
# KNN Model Training

knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) based on your dataset
knn_classifier.fit(X_train, y_train)
accuracy = accuracy_score(y_test, predictions)








@app.route('/', methods=['GET', 'POST'])
      
def home():
    if request.method == 'POST':

        symptom = request.form.get('text')
        preprocessed_symptom = preprocess_text(symptom)
        symptom_tfidf = tfidf_vectorizer.transform([preprocessed_symptom])
        predicted_disease = knn_classifier.predict(symptom_tfidf)
        # print(preprocessed_symptom)
        pred_disease = f'Predicted Disease: {predicted_disease[0]}'

        words = symptom.split(",")

        data1['common_words'] = data1['English_Symptoms'].apply(lambda x: sum(word.lower() in x.lower() for word in words))
        # Filter the data1 based on text similarity
        filtered_data = data1[data1['common_words'] > 0]

        # Sort the DataFrame based on the number of common words
        filtered_data = filtered_data.sort_values(by='common_words', ascending=False)

        # Drop the 'common_words' column as it's no longer needed
        filtered_data = filtered_data.drop(columns=['common_words'])

        original_data_same_indices = data1.loc[filtered_data.index]

        # Print or return the data1
        print(original_data_same_indices)
        ############################################################################################################################
        original_data_same_indices = original_data_same_indices.head(10)


        formulations_lst = list(df1['Name of Medicine'])

        original_list = list(df1['Main Indications'])

        processed_list = []

        for item in original_list:
            # Remove spaces and newline characters, convert to lowercase
            processed_item = ''.join(item.split()).lower()
            processed_list.append(processed_item)

        # print(processed_list[:5])

        # List of lists of symptoms
        list_of_symptoms = processed_list

        # Flatten the list of lists and split the symptoms using commas and spaces
        flat_symptoms = [symptom.replace(',', ' ').split() for symptoms in list_of_symptoms for symptom in symptoms.split(',')]

        # Get unique symptoms as a list
        unique_symptoms = list(set(symptom for sublist in flat_symptoms for symptom in sublist))

        # Print the unique symptoms
        # print(unique_symptoms[:5])

        data2 = {
            "Formulation": formulations_lst,
            "Symptoms": processed_list,
        }

        # Create a DataFrame
        df = pd.DataFrame(data2)
        symptoms['Symptom'] = symptoms['Symptom'].str.lower()

        correct_words = unique_symptoms

        data2 = {
            "Formulation": formulations_lst,
            "Symptoms": processed_list,
        }

        df = pd.DataFrame(data2)

        # Create a TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Transform the symptom text data2 into numerical features
        X_tfidf = tfidf_vectorizer.fit_transform(df['Symptoms'])

        # Create and train a classifier (e.g., Naive Bayes)
        clf = MultinomialNB()
        clf.fit(X_tfidf, df['Formulation'])

        # Spelling Correction
        user_input = get_column_values(original_data_same_indices, 'Symptom')
        print(user_input)
        input_symptoms = user_input.split()
        new_symptoms = correct_symptoms(input_symptoms)
       

        # Find Symptom Description
        symptoms_lst_desc(new_symptoms)

        # Predict Formulation
        new_symptoms_tfidf = tfidf_vectorizer.transform(new_symptoms)
        predicted_label = clf.predict(new_symptoms_tfidf)

        c = len(original_data_same_indices) if len(original_data_same_indices)<10 else 10
        medicines = []
        while (c>0):
            meds = []
            ### Create a boolean mask to filter rows where the second column matches any element in closest_formulations
            mask = df1.iloc[:, 0].isin([predicted_label[len(original_data_same_indices)-c]])
            # Use the mask to select the rows that match the condition
            filtered_df = df1[mask]

            # Iterate through the filtered DataFrame and print each row separately
            for index, row in filtered_df.iterrows():
                meds.append(row)
            c-=1
            medicines.append(meds)
        return render_template('index.html',
                                pred_disease,
                                original_data_same_indices,
                                medicines,
                                accuracy)

    extracted_text = request.args.get('text', '')

    return render_template('index.html', extracted_text=extracted_text)
    #return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)







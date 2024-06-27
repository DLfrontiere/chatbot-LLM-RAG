import os
import pandas as pd
import pickle

def create_or_update_csv(prompt, answer,groundtruth,context, answer_time, model_name, embeddings, retriever, pre_summarize, vectorstore, csv_file="./chat_history.csv"):
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        # Create a new DataFrame and save it as a new CSV file
        df = pd.DataFrame(columns=['prompt', 'answer','groundtruth','context', 'model', 'answer_time', 'embeddings', 'retriever', 'pre_summarize', 'vectorstore'])
        df.to_csv(csv_file, index=False)

    # Load the existing CSV file
    df = pd.read_csv(csv_file)
    
    context_strings = tuple(doc.page_content for doc in context)
    unique_context = ",".join(context_strings)

    # Create a new DataFrame for the new data
    new_data = pd.DataFrame([{
        'prompt': prompt,
        'answer': answer,
        'groundtruth':groundtruth,
        'context': unique_context,
        'model': model_name,
        'answer_time': answer_time,
        'embeddings': embeddings,
        'retriever': retriever,
        'pre_summarize': pre_summarize,
        'vectorstore': vectorstore
    }])

    # Concatenate the new data with the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)

def save_object(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
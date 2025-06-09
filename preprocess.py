import pandas as pd
import json
from transformers import BertTokenizer
import os

def load_squad_data(file_path):
    with open(file_path, 'r') as f:
        squad_data = json.load(f)
    return squad_data['data']

def preprocess_squad(data, output_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    contexts, questions, answers, answer_starts = [], [], [], []

    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer['text'])
                    answer_starts.append(answer['answer_start'])

    # Create DataFrame
    df = pd.DataFrame({
        'context': contexts,
        'question': questions,
        'answer': answers,
        'answer_start': answer_starts
    })

    # Data cleaning: remove rows with empty or overly long contexts/questions
    df = df[df['context'].str.len() > 0]
    df = df[df['context'].str.len() < 512]  # BERT max length
    df = df[df['question'].str.len() > 0]
    df = df.dropna()

    # Tokenization
    def tokenize_row(row):
        inputs = tokenizer(
            row['question'],
            row['context'],
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze().tolist(),
            'attention_mask': inputs['attention_mask'].squeeze().tolist(),
            'token_type_ids': inputs['token_type_ids'].squeeze().tolist()
        }

    tokenized_data = df.apply(tokenize_row, axis=1, result_type='expand')
    df = pd.concat([df, tokenized_data], axis=1)

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    print(f"Data quality improved by filtering {100 - (len(df) / len(contexts) * 100):.2f}%")

if __name__ == "__main__":
    squad_data = load_squad_data('../data/train-v1.1.json')
    preprocess_squad(squad_data, '../data/preprocessed_squad.csv')
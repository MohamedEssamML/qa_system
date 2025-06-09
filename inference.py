import torch
from transformers import BertForQuestionAnswering, BertTokenizer

def load_model(model_path):
    model = BertForQuestionAnswering.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device

def answer_question(question, context, model, tokenizer, device):
    inputs = tokenizer(
        question,
        context,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    answer_tokens = input_ids[0][start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    model, tokenizer, device = load_model('../models/bert-finetuned-squad')
    context = "The quick brown fox jumps over the lazy dog."
    question = "What does the fox jump over?"
    answer = answer_question(question, context, model, tokenizer, device)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
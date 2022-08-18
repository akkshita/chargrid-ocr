import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

t5_model = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(t5_model)


## Getting the Question representation
class Question_tokenization():
    def __init__(self, tokenizer, pad_token_id = 0, max_sequence_length = 512):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length
        
    #question input as string
    def foward(self, question):
        question_array = []
        question = question.split(" ")
  
        for token in question:
            question_array.extend(tokenizer(token, add_special_tokens = False).input_ids)
  
        if len(question_array)< self.max_sequence_length:
            question_array.extend([self.pad_token_id]* (self.max_sequence_length-len(question_array)))

        question_array = torch.tensor(question_array, dtype = torch.int32)
        return question_array[:self.max_sequence_length]

def convert_token_to_ques(ques, tokenizer):
  decoded_ques = tokenizer.decode(ques, skip_special_tokens=True)
  return decoded_ques
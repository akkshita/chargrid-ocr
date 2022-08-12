import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image, ImageDraw

from dataset import load_json_file, get_specific_file, resize_align_bbox, get_tokens_with_boxes, create_features
from modeling import LaTr_for_pretraining, LaTr_for_finetuning
from utils import convert_ans_to_token, convert_ques_to_token, rotate, convert_token_to_ques, convert_token_to_answer


class TextVQA(Dataset):
  def __init__(self, base_img_path, csv_df, ocr_json_df, tokenizer, transform = None, max_seq_length = 512, target_size = (512,512), fine_tune = True):

    self.base_img_path = base_img_path 
    # self.json_df = json_df
    self.csv_df = csv_df
    self.ocr_json_df = ocr_json_df
    self.tokenizer = tokenizer
    self.target_size = target_size
    self.transform = transform
    self.max_seq_length = max_seq_length
    self.fine_tune = fine_tune

  def __len__(self):
    return len(self.csv_df)

  def __getitem__(self, idx):

    # curr_img = self.json_df.iloc[idx]['image_id']
    curr_img = self.csv_df.iloc[idx]['ucsf_document_id']
    # ocr_token = self.ocr_json_df[self.ocr_json_df['image_id']==curr_img]['ocr_info'].values.tolist()[0]
    ocr_token = os.path.join(self.ocr_json_df, curr_img+".csv")

    boxes = []
    words = []

    current_group = self.json_df.iloc[idx]
    width, height = current_group['image_width'], current_group['image_height']

    ## Getting the ocr and the corresponding bounding boxes
    ocr_token_pandas = pd.read_csv(ocr_token)
    for i, row in ocr_token_pandas.iterrows:
      boundingbox = row['Bounding Boxes']
      text = row['Text']

      curr_bbox = [boundingbox[0], boundingbox[1], boundingbox[4], boundingbox[5]]
      boxes.append(curr_bbox)
      words.append(text)

    img_path = os.path.join(self.base_img_path, curr_img)+'.png'  ## Adding .jpg at end of the image, as the grouped key does not have the extension format 

    assert os.path.exists(img_path)==True, f'Make sure that the image exists at {img_path}!!'
    ## Extracting the feature
    
    if self.fine_tune:
        
        ## For fine-tune stage, they use [0, 0, 1000, 1000] for all the bounding box
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.target_size)
        boxes = torch.zeros(self.max_seq_length, 4)
        boxes[:, 2] = 1000
        boxes[:, 3] = 1000
        
        words = " ".join(words)
        tokenized_words = self.tokenizer.encode(words, max_length = self.max_seq_length, truncation = True, padding = 'max_length', return_tensors = 'pt')[0]
        
    else:
        ## For pre-train, this strategy would be useful
        img, boxes, tokenized_words = create_features(image_path = img_path,
                                                      tokenizer = self.tokenizer,
                                                      target_size = self.target_size,
                                                      max_seq_length = self.max_seq_length,
                                                      use_ocr = False,
                                                      bounding_box = boxes,
                                                      words = words
                                                      )
    
    ## Converting the boxes as per the format required for model input
    boxes = torch.as_tensor(boxes, dtype=torch.int32)
    width = (boxes[:, 2] - boxes[:, 0]).view(-1, 1)
    height = (boxes[:, 3] - boxes[:, 1]).view(-1, 1)
    boxes = torch.cat([boxes, width, height], axis = -1)

    ## Clamping the value,as some of the box values are out of bound
    boxes[:, 0] = torch.clamp(boxes[:, 0], min = 0, max = 1000)
    boxes[:, 2] = torch.clamp(boxes[:, 2], min = 0, max = 1000)
    boxes[:, 4] = torch.clamp(boxes[:, 4], min = 0, max = 1000)
    
    boxes[:, 1] = torch.clamp(boxes[:, 1], min = 0, max = 1000)
    boxes[:, 3] = torch.clamp(boxes[:, 3], min = 0, max = 1000)
    boxes[:, 5] = torch.clamp(boxes[:, 5], min = 0, max = 1000)
    
    ## Tensor tokenized words
    tokenized_words = torch.as_tensor(tokenized_words, dtype=torch.int32)

    if self.transform is not None:
      img = self.transform(img)
    else:
      img = transforms.ToTensor()(img)


    ## Getting the Question
    # question = current_group['question']
    question = self.csv_df.iloc[idx]['question']
    question = convert_ques_to_token(question = question, tokenizer = self.tokenizer)

    ## Getting the Answer
    # answer = current_group['answers']
    answer = self.csv_df.iloc[idx]['answers']
    answer = convert_ques_to_token(question = answer, tokenizer = self.tokenizer).long()

    return {'img':img, 'boxes': boxes, 'tokenized_words': tokenized_words, 'question': question, 'answer': answer, 'id': torch.as_tensor(idx)}


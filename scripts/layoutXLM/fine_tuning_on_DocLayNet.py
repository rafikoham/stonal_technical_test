import numpy as np
import collections
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt

from operator import itemgetter
from PIL import Image, ImageDraw, ImageFont
#from google.colab.patches import cv2_imshow # Colab
from ipywidgets import widgets
from IPython.display import display, HTML
from datasets import  concatenate_datasets

font = ImageFont.load_default()


########## datasets: 
# categories colors
label2color = {
    'Caption': 'brown',
    'Footnote': 'orange',
    'Formula': 'gray',
    'List-item': 'yellow',
    'Page-footer': 'red',
    'Page-header': 'red',
    'Picture': 'violet',
    'Section-header': 'orange',
    'Table': 'green',
    'Text': 'blue',
    'Title': 'pink'
    }

domains = ["Financial Reports", "Manuals", "Scientific Articles", "Laws & Regulations", "Patents", "Government Tenders"]
domain_names = [domain_name.lower().replace(" ", "_").replace("&", "and") for domain_name in domains]

# bounding boxes start and end of a sequence
cls_box = [0, 0, 0, 0]
sep_box = [1000, 1000, 1000, 1000]

# DocLayNet dataset
# dataset_name = "pierreguillou/DocLayNet-small"
dataset_name = "pierreguillou/DocLayNet-base"
dataset_name_suffix = dataset_name.replace("pierreguillou/DocLayNet-", "")

# parameters for tokenization and overlap
max_length = 384 # The maximum length of a feature (sequence)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.

# PAD token index
label_pad_token_id = -100

# parameters de TrainingArguments
batch_size=8 # WARNING: change this value according to your GPU RAM
num_train_epochs=3
learning_rate=2e-5
per_device_train_batch_size=batch_size
per_device_eval_batch_size=batch_size*2
gradient_accumulation_steps=1
warmup_ratio=0.1
evaluation_strategy="steps"
eval_steps=300
save_steps=300 # eval_steps
save_total_limit=1
load_best_model_at_end=True
metric_for_best_model="f1"
report_to="tensorboard"
fp16=True
push_to_hub=True # we'd like to push our model to the hub during training
hub_private_repo=True
hub_strategy="all_checkpoints"

# model name in HF
version = 1 # version number

output_dir = "DocLayNet/layout-xlm-base-finetuned-" + dataset_name.replace("pierreguillou/", "") + "_lines_ml" + str(max_length) + "-v" + str(version)
hub_model_id = "pierreguillou/layout-xlm-base-finetuned-" + dataset_name.replace("pierreguillou/", "") + "_lines_ml" + str(max_len)+"-v" + str(version)



# it is important that each bounding box should be in (upper left, lower right) format.
# source: https://github.com/NielsRogge/Transformers-Tutorials/issues/129
def upperleft_to_lowerright(bbox):
    x0, y0, x1, y1 = tuple(bbox)
    if bbox[2] < bbox[0]:
        x0 = bbox[2]
        x1 = bbox[0] 
    if bbox[3] < bbox[1]:
        y0 = bbox[3]
        y1 = bbox[1] 
    return [x0, y0, x1, y1]
  
# convert boundings boxes (left, top, width, height) format to (left, top, left+widght, top+height) format. 
def convert_box(bbox):
    x, y, w, h = tuple(bbox) # the row comes in (left, top, width, height) format
    return [x, y, x+w, y+h] # we turn it into (left, top, left+widght, top+height) to get the actual box 

# LiLT model gets 1000x10000 pixels images
def normalize_box(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


# LiLT model gets 1000x10000 pixels images
def denormalize_box(bbox, width, height):
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width* (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000)),
    ]

# get back original size
def original_box(box, original_width, original_height, coco_width, coco_height):
    return [
        int(original_width * (box[0] / coco_width)),
        int(original_height * (box[1] / coco_height)),
        int(original_width * (box[2] / coco_width)),
        int(original_height* (box[3] / coco_height)),
    ]


def get_blocks(bboxes_block, categories, texts):

 # get list of unique block boxes
    bbox_block_dict, bboxes_block_list, bbox_block_prec = dict(), list(), list()
    for count_block, bbox_block in enumerate(bboxes_block):
        if bbox_block != bbox_block_prec:
            bbox_block_indexes = [i for i, bbox in enumerate(bboxes_block) if bbox == bbox_block]
            bbox_block_dict[count_block] = bbox_block_indexes
            bboxes_block_list.append(bbox_block)
        bbox_block_prec = bbox_block

    # get list of categories and texts by unique block boxes
    category_block_list, text_block_list = list(), list()
    for bbox_block in bboxes_block_list:
        count_block = bboxes_block.index(bbox_block)
        bbox_block_indexes = bbox_block_dict[count_block]
        category_block = np.array(categories, dtype=object)[bbox_block_indexes].tolist()[0]
        category_block_list.append(category_block)
        text_block = np.array(texts, dtype=object)[bbox_block_indexes].tolist()
        text_block = [text.replace("\n","").strip() for text in text_block]
        if id2label[category_block] == "Text" or id2label[category_block] == "Caption" or id2label[category_block] == "Footnote":
            text_block = ' '.join(text_block)
        else:
            text_block = '\n'.join(text_block)
        text_block_list.append(text_block)

    return bboxes_block_list, category_block_list, text_block_list


# function to sort bounding boxes
def get_sorted_boxes(bboxes):

    # sort by y from page top to bottom 
    sorted_bboxes = sorted(bboxes, key=itemgetter(1), reverse=False)
    y_list = [bbox[1] for bbox in sorted_bboxes]

    # sort by x from page left to right when boxes with same y
    if len(list(set(y_list))) != len(y_list):
        y_list_duplicates_indexes = dict()
        y_list_duplicates = [item for item, count in collections.Counter(y_list).items() if count > 1]
        for item in y_list_duplicates:
            y_list_duplicates_indexes[item] = [i for i, e in enumerate(y_list) if e == item]
            bbox_list_y_duplicates = sorted(np.array(sorted_bboxes, dtype=object)[y_list_duplicates_indexes[item]].tolist(), key=itemgetter(0), reverse=False)
            np_array_bboxes = np.array(sorted_bboxes)
            np_array_bboxes[y_list_duplicates_indexes[item]] = np.array(bbox_list_y_duplicates)
            sorted_bboxes = np_array_bboxes.tolist()

    return sorted_bboxes


# sort data from y = 0 to end of page (and after, x=0 to end of page when necessary)
def sort_data(bboxes, categories, texts):

    sorted_bboxes = get_sorted_boxes(bboxes)
    sorted_bboxes_indexes = [bboxes.index(bbox) for bbox in sorted_bboxes]
    sorted_categories = np.array(categories, dtype=object)[sorted_bboxes_indexes].tolist()
    sorted_texts = np.array(texts, dtype=object)[sorted_bboxes_indexes].tolist()

    return sorted_bboxes, sorted_categories, sorted_texts


# get PDF image and its data
def generate_annotated_image(index_image=None, split="all"):

    # get dataset
    example = dataset

    # get split
    if split == "all":
        example = concatenate_datasets([example["train"], example["validation"], example["test"]])
    else:
        example = example[split]

    # get random image & PDF data
    if index_image == None: index_image = random.randint(0, len(example)-1)
    example = example[index_image]
    image = example["image"] # original image
    coco_width, coco_height = example["coco_width"], example["coco_height"]
    original_width, original_height = example["original_width"], example["original_height"]
    original_filename = example["original_filename"]
    page_no = example["page_no"]
    num_pages = example["num_pages"]

    # resize image to original
    image = image.resize((original_width, original_height))

    # get corresponding annotations
    texts = example["texts"]
    bboxes_block = example["bboxes_block"]
    bboxes_line = example["bboxes_line"]
    categories = example["categories"]
    domain = example["doc_category"]

    # get domain name
    index_domain = domain_names.index(domain)
    domain = domains[index_domain]

    # convert boxes to original
    original_bboxes_block = [original_box(convert_box(box), original_width, original_height, coco_width, coco_height) for box in bboxes_block]
    original_bboxes_line = [original_box(convert_box(box), original_width, original_height, coco_width, coco_height) for box in bboxes_line]

    ##### block boxes #####

    # get unique blocks and its data
    bboxes_blocks_list, category_block_list, text_block_list = get_blocks(original_bboxes_block, categories, texts)

    # sort data from y = 0 to end of page (and after, x=0 to end of page when necessary)
    sorted_original_bboxes_block_list, sorted_category_block_list, sorted_text_block_list = sort_data(bboxes_blocks_list, category_block_list, text_block_list)

    ##### line boxes ####

    # sort data from y = 0 to end of page (and after, x=0 to end of page when necessary)
    sorted_original_bboxes_line_list, sorted_category_line_list, sorted_text_line_list = sort_data(original_bboxes_line, categories, texts)

    # group paragraphs and lines outputs
    sorted_original_bboxes = [sorted_original_bboxes_block_list, sorted_original_bboxes_line_list]
    sorted_categories = [sorted_category_block_list, sorted_category_line_list]
    sorted_texts = [sorted_text_block_list, sorted_text_line_list]

    # get annotated boudings boxes on images
    images = [image.copy(), image.copy()]

    imgs, df_paragraphs, df_lines = dict(), pd.DataFrame(), pd.DataFrame()
    for i, img in enumerate(images):

        img = img.convert('RGB') # Convert to RGB
        draw = ImageDraw.Draw(img)
        
        for box, label_idx, text in zip(sorted_original_bboxes[i], sorted_categories[i], sorted_texts[i]):
            label = id2label[label_idx]
            color = label2color[label]
            draw.rectangle(box, outline=color)
            text = text.encode('latin-1', 'replace').decode('latin-1') # https://stackoverflow.com/questions/56761449/unicodeencodeerror-latin-1-codec-cant-encode-character-u2013-writing-to
            draw.text((box[0] + 10, box[1] - 10), text=label, fill=color, font=font)

        if i == 0: 
            imgs["paragraphs"] = img
        
            df_paragraphs["paragraphs"] = list(range(len(sorted_original_bboxes_block_list)))
            df_paragraphs["categories"] = [id2label[label_idx] for label_idx in sorted_category_block_list]
            df_paragraphs["texts"] = sorted_text_block_list
            df_paragraphs["bounding boxes"] = [str(bbox) for bbox in sorted_original_bboxes_block_list]

        else: 
            imgs["lines"] = img

            df_lines["lines"] = list(range(len(sorted_original_bboxes_line_list)))
            df_lines["categories"] = [id2label[label_idx] for label_idx in sorted_category_line_list]
            df_lines["texts"] = sorted_text_line_list
            df_lines["bounding boxes"] = [str(bbox) for bbox in sorted_original_bboxes_line_list]

    return imgs, original_filename, page_no, num_pages, domain, df_paragraphs, df_lines



    # display PDF image and its data
def display_pdf_blocks_lines(index_image=None, split="all"):

    # get image and image data
    images, original_filename, page_no, num_pages, domain, df_paragraphs, df_lines = generate_annotated_image(index_image=index_image, split=split)

    print(f"PDF: {original_filename} (page: {page_no+1} / {num_pages}; domain: {domain})\n")

    # left widget
    style1 = {'overflow': 'scroll' ,'white-space': 'nowrap', 'width':'50%'}
    output1 = widgets.Output(description = "PDF image with bounding boxes of paragraphs", style=style1)
    with output1:
    
        # display image
        print(">> PDF image with bounding boxes of paragraphs\n")
        
        open_cv_image = np.array(images["paragraphs"]) # PIL to cv2
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        # cv2.imshow('',open_cv_image) # lambda
        cv2_imshow(open_cv_image) # Colab
        cv2.waitKey(0)

        # display DataFrame
        print("\n>> Paragraphs dataframe\n")
        display(df_paragraphs)

    # right widget
    style2 = style1
    output2 = widgets.Output(description = "PDF image with bounding boxes of lines", style=style2)
    with output2:
    
        # display image
        print(">> PDF image with bounding boxes of lines\n")
  
        open_cv_image = np.array(images["lines"]) # PIL to cv2
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        # cv2.imshow('',open_cv_image) # lambda
        cv2_imshow(open_cv_image) # Colab
        cv2.waitKey(0)

        # display DataFrame
        print("\n>> Lines dataframe\n")
        display(df_lines)

    ## Side by side thanks to HBox widgets
    sidebyside = widgets.HBox([output1,output2])
    ## Finally, show.
    display(sidebyside)

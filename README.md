# stonal_technical_test





# Ressources : Blog, Papiers, Code (Repo)
## *Blogs: 
* https://huggingface.co/blog/document-ai

## *Document AI
### Document Image classification
#### Classifying docs into the approppriate category : A basic approach is to apply OCR on a document image, after which a BERT-like model is used for classification
#### Document Layout Analysis : LayoutML, Donut come into play by incorporating text and visual information

### Document layout analysis 
#### Doc layout analysis is the task of determining the physical structure of the document, like the individual blocks that make up a doc: text segments, headers and tables ==> State of art models are : LayoutLMv3 and DiT : Use Mask R-CNN framework for object dectection backbone.

### Document parsing
#### Document parsing is identifying and extracting key information from a document 

## *Code:
### Coding with Langchain : 
* https://github.com/praj2408/Langchain-PDF-App-GUI
* https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
### Transformers : 
* https://github.com/NielsRogge/Transformers-Tutorials

## Papers: 
* ### Donut : 
![Alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/donut_architecture.jpg)

Fine tuning tuto:  https://www.freecodecamp.org/news/how-to-fine-tune-the-donut-model/
* ### LayoutLMv3
![Alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/layoutlmv3_architecture.png)
import os

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from typing_extensions import Concatenate


os.environ["OPENAI_API_KEY"]="une api key ici"


def return_texts_from_pdf(
        pdf_file_location:str,
        splitter_seperator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,

):
    """
    Args:
        pdf_file_location : path du fichier pdf
    Returns 
        Splitted Text content (not metadata)
    """

    RAW_TEXT=''
    #lecture du pdf
    pdf_reader=PdfReader(pdf_file_location)
    for _, page in enumerate(pdf_reader.pages):
        content=page.extract_text()
        if content:
            RAW_TEXT+=content

    text_splitter=CharacterTextSplitter(
        separator=splitter_seperator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )
    texts=text_splitter.split_text(RAW_TEXT)

    return texts


def return_metadata_from_pdf(
        pdf_file_location:str
):
    """
    Args:
        pdf_file_location:
    Returns:
        the metadata as a list of texts
    """
    loader = PyMuPDFLoader(pdf_file_location)
    meta_data_dict_list=[]

    data = loader.load() 
    for i in range(len(data)):
        meta_data_dict_list.append(data[i].dict())

    new_dict={str(i):d for i,d in enumerate(meta_data_dict_list)}
    #Je supprime la page
    for i in range (len(data)):
                    del new_dict[f"{i}"]['metadata']['page']
    #the new dict now : 
    metadata_text=new_dict['0']['metadata']
    formatted_metadata = {key: f"{key} is {value}" if isinstance(value, str) else f"{key} is {str(value)}" for key, value in metadata_text.items()}
    texts = [value for key, value in formatted_metadata.items() if isinstance(value, str)]

    return texts
    

def query_pdf(
        query:str,
        texts:list,
):
    """
    Query: la question ou une informations à poser sur le pdf
    texts: liste des textes retournés de la func précédentes
    """
    embeddings=OpenAIEmbeddings()
    document_search=FAISS.from_texts(texts,embeddings)

    chain=load_qa_chain(OpenAI(),
                        chain_type="stuff")
    docs=document_search.similarity_search(query)
    reponse=chain.run(input_documents=docs,
              question=query)
    
    return reponse
    
    
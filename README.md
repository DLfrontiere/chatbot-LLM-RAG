# chatbot-LLM-RAG

Main types of RAG architectures:

![alt text](https://github.com/DLfrontiere/chatbot-LLM-RAG/blob/main/images/RAG_architectures.png?raw=True)

An example of chat bot LLM-RAG architecture:

![alt text](https://github.com/DLfrontiere/chatbot-LLM-RAG/blob/main/images/rag-chatbot-architecture-1.png?raw=true)

In particular in this repository is possible to:

- Load different types of files [pdf,txt,word,doc,html,docx] to be retrieved (use --file_path to specify the path within the files and modify accepted files variable to filter)
- Load different urls to be retrieved (variable inside code)
- Choose a model (--model ),for now supported open ai,groq and claude models,specify the api keys in api_key.env file
- Choose if summarize everything using the chosen model using --pre_summarize
- Choose an embedding model --embeddings hugging face,opeai and fast
- Chroma is used for vector storing //
  
- Choose a retriever using --retriever between base retreiever,parent document retriever and three type of ContextualCompressionRetriever(compressor,extractor,filter)
- The chatbot is chat history aware

# Usage  Example

python3 main.py --model openai --embeddings fast --retriever parent --files_path ./your_files_dir --pre_summarize False

A link will be generated to use a gui to interact wiht the chatbot


# Possible improvements
  
 other files can be used(images)? (images can also be etracted from the files like PDF) enable files upload for each user? store files in local or cloud? saving the loaded files for each user/chat?

local or cloud database? store every chat history?

 free/opensource models?fine tuning?efficiency?speed?cost? model able to read images?mobel able to generate images/tables/code?

 self evaluation?refine on top chunks? hierarchical summarization of context?multiple query generation with re-ranking?

 Multi agent approach?


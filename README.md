# chatbot-LLM-RAG

Main types of RAG architectures:

![alt text](https://github.com/DLfrontiere/chatbot-LLM-RAG/blob/main/images/RAG_architectures.png?raw=True)

An example of chat bot LLM-RAG architecture:

![alt text](https://github.com/DLfrontiere/chatbot-LLM-RAG/blob/main/images/rag-chatbot-architecture-1.png?raw=true)

In particular in this repository is possible to:

- Load different types of files [pdf,txt,word,doc,html,docx] to be retrieved (use --file_path to specify the path within the files and modify accepted files variable to filter)
- Load different urls to be retrieved (variable inside code).
- Choose a model family(--model ),for now supported open ai,groq,claude and google models,specify the api keys in api_key.env file.
- Choose a specific model (--model_name) fo the model family chosen.
- Choose if summarize everything using the chosen model using --pre_summarize.
- Choose which text splitter use (--splitter) [recursive,semantic].
- Choose an embedding model (--embeddings) hugging face,opeai,fast and google.
- Choose the vectorstore (--vectorstore) [qdrant,chroma,google].
- Choose a retriever using --retriever between base retriever(vectorstore),parent document retriever,multi query retriever and three type of ContextualCompressionRetriever(compressor,extractor,filter).
- The chatbot is chat history aware,


# Usage  Example
```python 
python3 main.py --model openai --model_name gpt-4o --embeddings fast --retriever parent --files_path ./your_files_dir --pre_summarize False --vectorstore qdrant --splitter semantic
```

The only one with no defualt value is file_path

```python 
python3 main.py --file_path your/path/to/files
```

You can uncomment the code line 
```python 
#GUI(answer_generator) 
```
In this way at the end of run will be generated a link to interact with the chatbot.

Otherwise you can write in the prompts list your queries and groundtruths (just for test) and a csv will be generated with queries and answers.




# Possible improvements
  
 other files can be used(images)? (images can also be etracted from the files like PDF) enable files upload for each user? store files in local or cloud? saving the loaded files for each user/chat?

local or cloud database? store every chat history?

 free/opensource models?fine tuning?efficiency?speed?cost? model able to read images?mobel able to generate images/tables/code?

 self evaluation?refine on top chunks? hierarchical summarization of context?multiple query generation with re-ranking?

 Multi agent approach?

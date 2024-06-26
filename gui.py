import gradio as gr
import pandas as pd
import os

class GUI():

    def __init__(self, answer_generator):
        self.answer_generator = answer_generator
        self.csv_file = './chat_history.csv'
        self.start_chat()

    def start_chat(self):

        def chat(inp):
            user_input = inp
            if not user_input:
                return None
            answer = self.answer_generator.answer_prompt(user_input)
            context = self.answer_generator.get_context()  # Retrieve context for the current session
            self.update_csv(user_input, answer, context)
            return f"\n{answer}"

        interface = gr.Interface(
            fn=chat,
            inputs=gr.Textbox(placeholder="Type your question here..."),
            outputs=gr.Textbox(value=""),
            title="Chat with me",
            description="Ask me anything!",
            elem_id="chat-container",
        )

        # Display the interface
        interface.launch()

    def update_csv(self, prompt, answer, context):
        # Check if the CSV file exists
        if not os.path.exists(self.csv_file):
            # Create a new DataFrame and save it as a new CSV file
            df = pd.DataFrame(columns=['prompt', 'answer', 'context'])
            df.to_csv(self.csv_file, index=False)

        # Load the existing CSV file
        df = pd.read_csv(self.csv_file)
        
        context_strings = tuple(doc.page_content for doc in context)
        unique_context = ",".join(context_strings)

        # Create a new DataFrame for the new data
        new_data = pd.DataFrame([{'prompt': prompt, 'answer': answer, 'context': unique_context}])

        # Concatenate the new data with the existing DataFrame
        df = pd.concat([df, new_data], ignore_index=True)

        # Save the updated DataFrame back to the CSV file
        df.to_csv(self.csv_file, index=False)


        
    



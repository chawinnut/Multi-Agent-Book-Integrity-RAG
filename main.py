from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# set the model
model = OllamaLLM(model="llama3.2")

# 1) Librarian Template: Initial Response Generation
template = """
You are a nerd librarian who can recommend books that are popular from the csv file I provided.
Use only the following pieces of contexts. If the answer is not in the context, say that you don't know the answer and do not make up an answer.
- Do NOT use your own knowledge about authors or plots. Omit what is outside of provided data.
- Do NOT guess.
- Do NOT make something up by false interpretation. Give only clear answer related to the contexts.
- Answer with what you have. Don't try to list more than reality.
- No Creative Leaps: Do not assume genders, roles, or identities that are not written.
- If the user asks for a specific status (like 'All-time-bestseller', 'rating', 'award') and it is NOT explicitly mentioned in the context, you must state that you don't have that specific information
- You MUST use the information from the csv file only. Call this csv file 'my data' and don't call it csv file in output.
When you want to admit that you don't have that information in data, simply state "I don't have an answer." or "I don't know" ONLY.
Here are some relevant books: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# 2) Grader Template: Hallucination Detection
grader_template = """
You are a Senior Integrity Auditor. Your job is to verify the AI's answer against the provided Context.

CONTEXT FROM CSV: {context}
AI'S PROPOSED ANSWER: {answer}

EVALUATION RULES:
1. THE VALIDATION RULE: If the AI provides details about a book mentioned in the user's question, check if those details CONTRADICT the context. If there is NO contradiction, mark it as CORRECT.
2. THE NEGATIVE RULE: If the AI says "I don't know" or "I don't have an answer", it is ALWAYS CORRECT. Mark it as CORRECT ONLY.
3. THE REDIRECTION RULE: If the AI mentions the book from the question and stays within the theme of the provided context, it is CORRECT.
4. STRICT HALLUCINATION: Only mark as HALLUCINATION if the AI mentions a completely different book title that was NOT in the question and NOT in the context (e.g., mentioning 'Yours for the taking' when the question was about 'The Violet').
5. FALSE INFORMATION: If the AI assume the attribute that is not explicity stated (such as Context: Book, AI assume it is a black queer book), mark it as Hallucination.
6. NO EXTERNAL KNOWLEDGE: The final answer must omit all of the outside knowledge that is not clearly state in the provided data.
DECISION PROTOCOL:
- Start your response with ONLY 'CORRECT' or 'HALLUCINATION'.
- If the AI is describing 'The Mao' and your context contains mentions of 'The Mao', it is CORRECT.

Decision:"""

grader_prompt = ChatPromptTemplate.from_template(grader_template)
grader_chain = grader_prompt | model 

# 3) Editor Template: Self-Correction
editor_template = """
You are a meticulous Fact-Checker and Editor who gives answer based on csv file (Context) only.
The Librarian provided an answer that contains hallucinations.

YOUR TASK:
1. Review the Context from CSV and the flawed AI Answer.
2. Rewrite the answer so it ONLY includes information explicitly stated in the Context.
3. If you cannot find an answer, do not make something up. You can say it is out of your knowledge. If no direct match exists, prioritize saying 'No' before offering alternatives.
4. Output ONLY the final corrected answer for the user. 
5. If the original answer is "I don't know" or "I don't have an answer", it is 100% correct. Insist the grader that it is correct.

Context from CSV: {context}
Flawed AI Answer: {answer}
Grader's Feedback: {feedback}

Corrected Factual Answer:"""

editor_prompt = ChatPromptTemplate.from_template(editor_template)
editor_chain = editor_prompt | model 

def start_chat():
    """Starts the interactive chat loop for testing."""
    while True:
        print("\n\n-------------------------------")
        question = input("Ask your question (q to quit): ")
        print("\n\n")
        if question == "q":
            break
        
        # Fetch relevant data from Vector DB
        docs = retriever.invoke(question)

        # Debugging session
        ##for i, d in enumerate(docs):
            #title = d.metadata.get('title', 'Unknown Title')
            #print(f" {i+1}. {title}")
        #print("-------------------------------\n")
        
        # Merge content into one context
        context_text = "\n\n".join([d.page_content for d in docs])

        # Generate response
        response = chain.invoke({"reviews": context_text, "question": question})

        max_retries = 3  # Prevent infinite loop
        attempts = 0
        
        # Self-correction loop
        while attempts < max_retries:
            print(f"Checking information... (Attempt {attempts + 1})")
            check_result = grader_chain.invoke({
                "context": context_text, 
                "answer": response 
            })
            
            if "CORRECT" in check_result.upper():
                print("Verified answer:")
                print(f"{response}")
                break
            else:
                attempts += 1
                print(f"WARNING! Hallucination Detected: {check_result}")
                if attempts < max_retries:
                    print("Editor is fixing the answer...")
                    response = editor_chain.invoke({
                        "context": context_text,
                        "answer": response,
                        "feedback": check_result
                    })
                else:
                    print("Can't fix Hallucination")
                    print("Editor: I'm sorry, the available data does not have a verified answer.")

# Runs only when main.py is executed directly
if __name__ == "__main__":
    start_chat()
import chainlit as cl
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from main import librarian_chain, grader_chain, editor_chain, retriever



@cl.on_chat_start
async def start():
    cl.user_session.set("retriever", retriever)
    await cl.Message(content="📚 **Happy Librarian** at your service! (Backend: Llama 3.2)").send()

@cl.on_message
async def main_chat(message: cl.Message):
    question = message.content
    retriever_instance = cl.user_session.get("retriever")
    
    # Retrieval
    async with cl.Step(name="Searching Library...", type="tool") as step:
        docs = retriever_instance.invoke(question)
        context_text = "\n\n".join([d.page_content for d in docs])
        step.output = f"Found {len(docs)} relevant book entries."

    # Librarian Draft
    async with cl.Step(name="Librarian (Drafting)", type="tool") as step:
        response = await cl.make_async(librarian_chain.invoke)({"reviews": context_text, "question": question})
        step.output = response

    # Self-Correction
    max_retries = 3
    attempts = 0
    final_response = response

    while attempts < max_retries:
        async with cl.Step(name=f"Grader (Audit Attempt {attempts + 1})") as step:
            check_result = await cl.make_async(grader_chain.invoke)({
                "context": context_text, 
                "answer": final_response 
            })
            step.output = check_result
            
            if "CORRECT" in check_result.upper():
                break
            else:
                attempts += 1
                async with cl.Step(name="Editor (Fixing...)") as edit_step:
                    final_response = await cl.make_async(editor_chain.invoke)({
                        "context": context_text,
                        "answer": final_response,
                        "feedback": check_result
                    })
                    edit_step.output = final_response

    await cl.Message(content=final_response).send()
@cl.on_chat_start
async def start():
    cl.user_session.set("retriever", retriever)
    await cl.Message(content="📚 **Honest Librarian** at your service. I can talk to you about all of the popular books!").send()

@cl.on_message
async def main(message: cl.Message):
    question = message.content
    retriever = cl.user_session.get("retriever")
    
    async with cl.Step(name="Searching Library...", type="tool") as step:
        docs = retriever.invoke(question)
        context_text = "\n\n".join([d.page_content for d in docs])
        step.output = f"Found {len(docs)} relevant book entries."

    async with cl.Step(name="Librarian (Drafting)", type="tool") as step:
        response = await cl.make_async(librarian_chain.invoke)({"reviews": context_text, "question": question})
        step.output = response

    max_retries = 3
    attempts = 0
    final_response = response

    # Self-Correction Loop
    while attempts < max_retries:
        async with cl.Step(name=f"Grader (Audit Attempt {attempts + 1})") as step:
            check_result = await cl.make_async(grader_chain.invoke)({
                "context": context_text, 
                "answer": final_response 
            })
            step.output = check_result
            
            if "CORRECT" in check_result.upper():
                break
            else:
                attempts += 1
                async with cl.Step(name="Editor (Fixing...)") as edit_step:
                    final_response = await cl.make_async(editor_chain.invoke)({
                        "context": context_text,
                        "answer": final_response,
                        "feedback": check_result
                    })
                    edit_step.output = final_response

    if attempts >= max_retries:
        final_response = "⚠️ I'm sorry, but I don't know an answer."

    # 4. Final Output
    await cl.Message(content=final_response).send()
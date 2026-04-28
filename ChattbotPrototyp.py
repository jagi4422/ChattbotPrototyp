from typing import List, TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, convert_to_messages, RemoveMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from pydantic import Field
import operator


#State där minnet och konversationshistoriken sparas
class State(MessagesState):
    messages: List
    summary: str 

#Välj modell för embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-nomic-embed-text-v1.5",
    openai_api_key="lm-studio",
    openai_api_base = "http://127.0.0.1:1234/v1"
)

#Vector-store, databas för RAG-systemet
memory_store = Chroma(
    collection_name="conversation_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_memory"
)

#Val av modell där denna drivs lokalt av LM Studio, här går att välja 
# nästan vilken (text)-modell man vill som LM Studio kan köra. Men LangChain
# stödjer även andra hostingsystem 
model = ChatOpenAI(
    model="llama-3-8b-instruct", 
    openai_api_key="lm-studio", 
    openai_api_base="http://127.0.0.1:1234/v1"
)
thread_id = "chat1"

#Här kan en användarprofil definieras, i denna prototyp görs deta under körning
user_profile = {
    "name": None,
    "interests": set()
}

DEBUG = False
MAX_CONVO_LENGTH = 6

#FUNKTIONER -----------------------------------------------------------------
#Följande är logik för att spara en konversation till chroma för att senare kunna hämta denna
#Ej helt fungerande tyvärr

def summarize_full_conversation(messages):

    lines =[]
    for m in messages:
        role = "System" if isinstance(m, SystemMessage) else "Användare" if isinstance(m, HumanMessage) else "Assistent"
        lines.append(f"{role}: {m.content}")
    conversation_text = "\n".join(lines)
    if DEBUG:
        print(f"DEUBG: conversation_text:\n{conversation_text}")

    summary_prompt = [
        SystemMessage(content="Sammanfatta hela konversationen mellan Assistent och Användare kortfattat och tydligt med hjälp av sammanfattningen."),
        HumanMessage(content=conversation_text)
    ]
    if DEBUG:
        print(f"DEBUG: summary_prompt:\n{summary_prompt}")
    summary = model.invoke(summary_prompt).content
    if DEBUG:
        print(f"DEBUG:TYPE:{type(summary)} SUMMARY:\n{summary}")
    return summary


def save_summary_to_chroma(summary: str, thread_id: str):
    if DEBUG:
        print(f"DEBUG:SUMMARY SENT TO CHROMA:\n{summary}\n SUMMARY TYPE:{type(summary)}")
    memory_store.add_texts(
        texts=[summary],
        metadatas=[{'thread_id': thread_id}],
        ids=[f"summary_{thread_id}"]
    )

def load_summary_from_chroma(thread_id: str):
    results = memory_store.get(ids=f"summary_{thread_id}")
    if results and "documents" in results and results["documents"]:
        return results["documents"][0]
    return ""




# LANGGRAPH NODES NEDAN ---------------------------------------------------------------

def running_summary(state: State):
    
    
    #Summerar konversationen kontinuerligt när antalet meddelanden har uppnåt gränsen specificerad
    #i MAX_CONVO_LENGTH för att inte uppnå gränsen för kontextlängden för LLMen
    #Gör detta genom att summera och sedan kapa gamla meddelanden


    convo = ([m for m in state["messages"] if isinstance(m,HumanMessage) or isinstance(m,AIMessage)])
    if len(convo) <= MAX_CONVO_LENGTH:
        return None
    
    summary = state.get("summary","")

    if summary:
        prompt = (
            f"Detta är en summering av nuvarande konversationen:{summary}\n\n"
            "Förläng summeringen genom att summera de nya meddelandena ovan:"
        )
    else:
        prompt = "Skapa en summering av konversationen ovan och svara ENDAST med sammanfattningen:"

    messages = convo + [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    if DEBUG:
        print(f"\nDEBUG: Summering av meddelanden då meddelanden är fler än {MAX_CONVO_LENGTH}, antal nuvarande meddelanden={len(convo)}\n\n {response}\n\n")

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    if DEBUG:
        print(f"DEBUG: Summering följt av sista två meddelanden som är kvar:\n\n {summary}\n\n{delete_messages}")

    return {"summary":response.content, "messages": delete_messages}

def extract_user_info(state: State):
    
    #Väldigt enkel logik för att extrahera namn och intressen från användaren. Detta ör ett
    #exempel på hur modellen kan samla in information i pågående samtal för att kunna ge återkoppling
    #eller hålla sig till sin kontext.
    
    last_msg = state["messages"][-1].content

    #väldigt enkel check för namn och intressen
    if "jag heter" in last_msg.lower():
        name = last_msg.split("heter")[-1].strip()
        user_profile["name"] = name

    if "jag gillar" in last_msg.lower():
        interests = last_msg.split("gillar")[-1].strip().split(",")
        for i in interests:
            user_profile["interests"].add(i.strip())

    return state

PROFILE_PREFIX = "[ANVÄNDARPROFIL]"

def retrieve_user_info(state: State):
    
    #Sammanställer insamlande intressen från konversationen hittills. Går att sammanföra med
    #RAG-databsen för att skapa en mer långvarig profil för varje användare.
    
    facts = []

    if user_profile["name"]:
        facts.append(f"Användaren heter {user_profile['name']}.")

    if user_profile["interests"]:
        interests = ", ".join(user_profile["interests"])
        facts.append(f"Användaren gillar: {interests}.")

    profile_message = SystemMessage( content=PROFILE_PREFIX + " " + " ".join(facts) )

    replaced = False

    for i, msg in enumerate(state["messages"]):
        if isinstance(msg,SystemMessage) and msg.content.startswith(PROFILE_PREFIX):
            state["messages"][i] = profile_message
            replaced = True
            break

    if not replaced:
        state["messages"].insert(0, profile_message)
    if DEBUG:
        print(facts)
    return state



def chat_node(state: State):
    
    #Noden där alla systemmeddelanden, tidigare konversationshistorik och sammanfattning
    #skickas in till modellen för att generera ett svar
    
    if DEBUG:
        print("DEBUG: STATE VID chat_node:\n",state["messages"])

    system_message = SystemMessage(content="Du ska rollspela som en snäll kompis, var snäll och kortfattad.")

    earlierconvo = state.get("summary","")
    if earlierconvo:
        previous_conversation = SystemMessage(content=f"Sammanfattning av konversationen hittils:\n {earlierconvo}") 
        prompt = [system_message, previous_conversation]
    else:
        prompt = [system_message]

    history = state["messages"]
    prompt = prompt + history
    response = model.invoke(prompt)
    return {"messages": [response]}


# LANGGRAPH BUILD OCH COMPILE --------------------------------------------------

workflow = StateGraph(state_schema=MessagesState)

workflow.add_node("chat", chat_node)
workflow.add_node("extract", extract_user_info) 
workflow.add_node("retrieve", retrieve_user_info)
workflow.add_node("summarize",running_summary)

workflow.add_edge(START,"extract")
workflow.add_edge("extract", "retrieve") 
workflow.add_edge("retrieve","chat")
workflow.add_edge("chat", "summarize")

chat_app = workflow.compile(checkpointer= MemorySaver())

#KÖRNINGSLOOP -------------------------------------------------------------------

while True:
    user_input = input("Du: ")

    if user_input.lower() in ["exit","end"]:
        break
    
    state_update = {
        "messages": [HumanMessage(content=user_input)],
        }

    result = chat_app.invoke(
        state_update,
        {"configurable":{"thread_id": thread_id}}
    )

    print("Bot: ", result["messages"][-1].content,"\n\n")



# Efter att användaren skrivit 'exit' eller 'end'
#full_summary = summarize_full_conversation(result["messages"])
#save_summary_to_chroma(full_summary, thread_id)


#print("\n--- Sammanfattning av konversationen ---")
#print(full_summary)

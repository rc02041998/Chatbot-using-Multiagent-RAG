from vectorstore import get_vector_retriever
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class HRPolicyAgent:
    def __init__(self):
        self.retriever = get_vector_retriever("hr_policy_doc.txt")
        self.llm = OllamaLLM(model="llama3.2")
        prompt = ChatPromptTemplate.from_template(
            "Based on the following HR policy context, answer the question:\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        )
        combine_docs_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        self.qa_chain = create_retrieval_chain(retriever=self.retriever, combine_docs_chain=combine_docs_chain)
    
    def get_response(self, query: str):
        result = self.qa_chain.invoke({"input": query})
        return result["answer"]

class HRBenefitsAgent:
    def __init__(self):
        self.retriever = get_vector_retriever("hr_benefits_doc.txt")
        self.llm = OllamaLLM(model="llama3.2")
        prompt = ChatPromptTemplate.from_template(
            "Based on the following HR benefits context, answer the question:\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        )
        combine_docs_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        self.qa_chain = create_retrieval_chain(retriever=self.retriever, combine_docs_chain=combine_docs_chain)
    
    def get_response(self, query: str):
        result = self.qa_chain.invoke({"input": query})
        return result["answer"]

class MultiAgentRAG:
    def __init__(self):
        self.agents = {
            "policy": HRPolicyAgent(),
            "benefits": HRBenefitsAgent()
        }
    
    def route_query(self, query: str):
        self.classifier_llm = OllamaLLM(model="llama3.2")
        prompt = ChatPromptTemplate.from_template(
            "Classify the following user query as either 'policy' or 'benefits':\n\n"
            "Query: {input}\n\n"
            "INSTRUCTIONS:"
            '1. Keep the output strictly "policy" or "benefits" classes'
            "<output>class name</output>"
        )
        result = self.classifier_llm.invoke(prompt.format(input=query))
        print("CATEGORY-----",result)
        category = result.strip().lower()
        if '<output>' in category:
            category = category.replace('<output>', '')
        elif '</output>' in category:
            category = category.replace('</output>', '')
        category = result.strip().lower()
        category = 'policy' if 'policy' in category else 'benefits'
        print("CLEANED CATEGORY-----",category)
        return "policy" if category not in self.agents else category
        
    def get_response(self, query: str):
        agent_key = self.route_query(query)
        print("-----AGENT-----",agent_key)
        return self.agents[agent_key].get_response(query)
    
    def get_agents(self):
        return list(self.agents.keys())

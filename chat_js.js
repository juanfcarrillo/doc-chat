import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnableMap } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers"
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { BufferWindowMemory } from "langchain/memory";
import { BaseListChatMessageHistory } from "@langchain/core/chat_history";

const loader = new PDFLoader("docs/spaceport.pdf");

const loadedDocs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100,
});

const docs = await textSplitter.splitDocuments(loadedDocs);

const model = new ChatOpenAI({
  temperature: 0.9,
  modelName: "gpt-3.5-turbo",
  apiKey: ""
});


const embeddings = new OpenAIEmbeddings({
    apiKey: ""
});

// vector store and retriever

const vectorstore = await Chroma.fromDocuments(docs, embeddings, {
    collectionName: "test-docs",
    persistDirectory: "./data"
});

const retriever = vectorstore.asRetriever({
    searchKwargs: {
        k: 5
    }
});

const SYSTEM_TEMPLATE = `
Answer the user's questions based on the below context.
Respond in the same language as the user.

<context>
{context}
</context>
`
const prompt = ChatPromptTemplate.fromMessages([
  ['system', SYSTEM_TEMPLATE],
  new MessagesPlaceholder("chat_history"),
  ['human', '{input}'],
]);

function formatDocs(docs) {
    return docs.map((doc) => {
        return doc.pageContent;
    }).join("\n");
}

const parrallel = RunnableMap.from({
    context: async (obj) => {
        return formatDocs(await retriever.invoke(obj.input));
    },
    input: (obj) => {
        return obj.input;
    },
    chat_history: (obj) => {
        return obj.chat_history;
    }
})

function createChain(
    prompt,
    llm_model
) {
    return parrallel
    .pipe(prompt)
    .pipe(llm_model)
    .pipe(new StringOutputParser())
}

const chain = createChain(prompt, model);

class BufferChatWindowMemory extends BaseListChatMessageHistory {
    window_memory = new BufferWindowMemory({ k: 5, returnMessages: true });

    constructor() {
        super();
    }

    async getMessages() {
        return (await this.window_memory.loadMemoryVariables({})).history;
    }

    async addMessages(messages) {
        await this.window_memory.saveContext({ input: messages[0].content }, { output: messages[1].content });
    }

    async clear() {
        await this.window_memory.clear();
    }
}

const history = new BufferChatWindowMemory();

history.addMessages([
    {
        role: "system",
        content: "You are a helpful assistant."
    },
    {
        role: "user",
        content: "What is the purpose of the spaceport?"
    }
])

const historyChain = new RunnableWithMessageHistory({
    runnable: chain,
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    getMessageHistory: async (sessionId) => history
});

const response = await historyChain.invoke(
    {
        input: "What is the purpose of the spaceport?"
    },
    {
        configurable: {
            sessionId: "test",
        }
    }
)

console.log(response)
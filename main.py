import argparse
import os
import langgraph_agent

def main():
    parser = argparse.ArgumentParser(description='mag CLI - Interactive RAG Chat')
    parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--collection', default='bookstack_pages', help='Chroma collection name')
    parser.add_argument('--top-k', type=int, default=3, help='Top K results')
    parser.add_argument('--google-llm-model', default=None, help='Name of Google LLM to use for generation (optional)')
    parser.add_argument('--google-embedding-model', default=None, help='Name of Google embedding model to use (required by agent or set via GOOGLE_EMBEDDING_MODEL env var)')

    args = parser.parse_args()

    # Validate required models are present or set in environment
    if not (args.google_embedding_model or os.environ.get('GOOGLE_EMBEDDING_MODEL')):
        print('Error: embedding model required. Pass --google-embedding-model or set GOOGLE_EMBEDDING_MODEL in your environment')
        return
    if not (args.google_llm_model or os.environ.get('GOOGLE_LLM_MODEL')):
        print('Error: LLM model required. Pass --google-llm-model or set GOOGLE_LLM_MODEL in your environment')
        return
    
    try:
        agent = langgraph_agent.build_langgraph_rag_agent(
            args.persist_dir,
            args.collection,
            top_k=args.top_k,
            embedding_model=args.google_embedding_model,
            llm_model=args.google_llm_model,
        )
    except Exception as e:
        print('Failed to create agent:', e)
        return

    print('Welcome to mag - Interactive RAG Chat')
    print('Type "exit" to quit\n')

    while True:
        try:
            query = input('You: ').strip()
            
            if query.lower() == 'exit':
                print('Goodbye!')
                break
            
            if not query:
                continue
            
            # Invoke agent to get answer
            out = agent.invoke(query)
            answer = out.get('answer') if out else None
            if answer:
                print('\nAssistant:')
                print(answer)
            else:
                print('No answer generated; ensure GOOGLE_LLM_MODEL and GOOGLE_API_KEY environment variables are configured')
            
            print()
        except KeyboardInterrupt:
            print('\nGoodbye!')
            break
        except Exception as e:
            print(f'Error processing query: {e}\n')


if __name__ == "__main__":
    main()

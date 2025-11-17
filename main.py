import argparse
import langgraph_agent

def main():
    parser = argparse.ArgumentParser(description='mag CLI')
    subparsers = parser.add_subparsers(dest='command')

    rag_parser = subparsers.add_parser('rag', help='Run RAG using a LangGraph agent')
    rag_parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    rag_parser.add_argument('--collection', default='bookstack_pages', help='Chroma collection name')
    rag_parser.add_argument('--query', required=True, help='Query string to search against')
    rag_parser.add_argument('--top-k', type=int, default=3, help='Top K results')
    rag_parser.add_argument('--google-llm-model', default=None, help='Name of Google LLM to use for generation (optional)')
    rag_parser.add_argument('--google-embedding-model', default=None, help='Name of Google embedding model to use (required by agent or set via GOOGLE_EMBEDDING_MODEL env var)')
    rag_parser.add_argument('--no-gen', action='store_true', help='Skip generation; only show retrieved docs')
    # `--sample` removed: embedding/LLM models must be configured (via args or env)

    args = parser.parse_args()

    if args.command == 'rag':
        # embedding_model and llm_model can be passed or set in environment (GOOGLE_EMBEDDING_MODEL, GOOGLE_LLM_MODEL)
        # Validate required models are present or set in environment
        import os
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

        # Show retrieval results
        results = agent.similarity_search(args.query, k=args.top_k)
        if results:
            print('Top results:')
            for i, r in enumerate(results, start=1):
                if isinstance(r, dict):
                    text = r.get('content', '')
                    meta = r.get('metadata', {})
                else:
                    text = getattr(r, 'page_content', '')
                    meta = getattr(r, 'metadata', {})
                print('----')
                print(f'Result {i}')
                print('Metadata:', meta)
                print('Text snippet:', text[:300])
        else:
            print('No results found for query:', args.query)

        if args.no_gen:
            return

        out = agent.invoke(args.query)
        answer = out.get('answer') if out else None
        if answer:
            print('\n---\nGenerated answer:')
            print(answer)
        else:
            print('No answer generated; ensure GOOGLE_LLM_MODEL and GOOGLE_API_KEY environment variables are configured')
    else:
        print('Hello from mag! (use `rag` command to run RAG)')


if __name__ == "__main__":
    main()

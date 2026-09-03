"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared Cohere client.
"""

import sqlite3

import cohere
from dotenv import load_dotenv
from gitsource import GithubRepositoryDataReader
from minsearch import Index
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult, SpanProcessor
from rag_helper import INSTRUCTIONS, PROMPT_TEMPLATE, RAGBase

load_dotenv()


class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("DROP TABLE IF EXISTS spans")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = span.attributes or {}
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self, timeout_millis=30000):
        return True


# Custom span processor to collect spans for answers
class CollectingSpanProcessor(SpanProcessor):
    def __init__(self):
        self.spans = []

    def on_start(self, span, parent_context):
        pass

    def on_end(self, span):
        self.spans.append(span)

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


# OpenTelemetry setup
collector = CollectingSpanProcessor()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter("traces.db")))
provider.add_span_processor(collector)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)


class RAGCohere(RAGBase):
    """Cohere-compatible RAG implementation extending RAGBase."""

    def llm(self, prompt):
        input_messages = [
            {'role': 'system', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.chat(
            model=self.model,
            messages=input_messages
        )

        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)

        # Extract text content from response (handles different content item types)
        response_text = next((item.text for item in response.message.content if hasattr(item, "text")), "")
        return response_text


class RAGTraced(RAGCohere):
    """RAG implementation with OpenTelemetry tracing."""

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_results", num_results)
            return super().search(query, num_results)

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            span.set_attribute("model", self.model)
            span.set_attribute("prompt_length", len(prompt))

            response = super().llm(prompt)

            # Capture token usage from Cohere response
            if hasattr(response, 'usage') and response.usage:
                tokens = getattr(response.usage, 'tokens', None)
                if tokens:
                    span.set_attribute("input_tokens", tokens.input_tokens)
                    span.set_attribute("output_tokens", tokens.output_tokens)

                    # Calculate cost (Cohere pricing - approximately $0.15/1M input tokens, $0.60/1M output tokens)
                    input_cost = tokens.input_tokens * 0.15 / 1_000_000
                    output_cost = tokens.output_tokens * 0.60 / 1_000_000
                    total_cost = input_cost + output_cost
                    span.set_attribute("cost", total_cost)

            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            span.set_attribute("query", query)
            result = super().rag(query)
            span.set_attribute("answer_length", len(result))
            return result


# Initialize Cohere client and traced RAG instance
rag = RAGTraced(
    index=index,
    llm_client=cohere.ClientV2(),
    model='command-a-plus-05-2026'
)

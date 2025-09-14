"""
Phoenix tracing setup for observability and debugging
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def setup_phoenix_tracing(phoenix_endpoint: Optional[str] = None):
    """Setup Phoenix tracing for LLM observability"""
    
    if phoenix_endpoint is None:
        phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006")
    
    try:
        # Import Phoenix instrumentation
        from phoenix.otel import register
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        
        # Setup OTLP exporter
        otlp_endpoint = f"{phoenix_endpoint}/v1/traces"
        span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        span_processor = BatchSpanProcessor(span_exporter)
        
        # Register Phoenix tracer
        tracer_provider = register(
            project_name="agentic-rag",
            span_processors=[span_processor]
        )
        
        # Instrument LlamaIndex
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        
        logger.info(f"Phoenix tracing setup completed. Endpoint: {phoenix_endpoint}")
        
    except ImportError as e:
        logger.warning(f"Phoenix tracing dependencies not available: {e}")
    except Exception as e:
        logger.error(f"Failed to setup Phoenix tracing: {e}")

def setup_phoenix_for_crewai():
    """Setup Phoenix tracing specifically for CrewAI agents"""
    
    try:
        from phoenix.otel import register
        from openinference.instrumentation.crewai import CrewAIInstrumentor
        
        # Register Phoenix for CrewAI
        tracer_provider = register(project_name="agentic-rag-crewai")
        CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
        
        logger.info("Phoenix tracing for CrewAI setup completed")
        
    except ImportError:
        logger.warning("CrewAI Phoenix instrumentation not available")
    except Exception as e:
        logger.error(f"Failed to setup CrewAI Phoenix tracing: {e}")

def log_evaluation_metrics(metrics: dict, test_case_id: str = None):
    """Log evaluation metrics to Phoenix"""
    
    try:
        from phoenix.trace import trace
        
        with trace(
            span_name="ragas_evaluation",
            attributes={
                "test_case_id": test_case_id,
                "evaluation_type": "ragas",
                **{f"metric.{k}": v for k, v in metrics.items()}
            }
        ):
            logger.info(f"Logged evaluation metrics to Phoenix: {metrics}")
            
    except ImportError:
        logger.debug("Phoenix trace logging not available")
    except Exception as e:
        logger.error(f"Failed to log metrics to Phoenix: {e}")

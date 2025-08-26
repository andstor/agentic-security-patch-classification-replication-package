from collections import OrderedDict

from opentelemetry.sdk.trace.export import SpanProcessor

class TraceSpanTracker(SpanProcessor):
    def __init__(self):
        # Maps trace_id -> set of span_ids
        self.traces = OrderedDict()  # regular dict

    def on_start(self, span, parent_context):
        trace_id = span.context.trace_id
        span_id = span.context.span_id

        # Initialize set if trace_id not seen yet
        if trace_id not in self.traces:
            self.traces[trace_id] = set()
        
        self.traces[trace_id].add(span_id)

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        pass

    # ----------------------
    # New helper methods
    # ----------------------

    def clear(self):
        """Clear all stored traces and spans."""
        self.traces.clear()

    def get_last_trace(self, ):
        """Return the last trace"""
        if not self.traces:
            return None, []
        
        last_trace_id = next(reversed(self.traces))
        return last_trace_id, list(self.traces[last_trace_id])



def trace_id_to_hex(trace_id: int) -> str:
    """Convert a 128-bit trace_id int to 32-char hex string."""
    return f"{trace_id:032x}"

def span_id_to_hex(span_id: int) -> str:
    """Convert a 64-bit span_id int to 16-char hex string."""
    return f"{span_id:016x}"
